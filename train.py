import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE   = 224          # MobileNetV2 native size
BATCH_SIZE = 32
SEED       = 42


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune MobileNetV2 for plant disease detection")
    p.add_argument("--data_dir",    type=str, default="./dataset",  help="Path to dataset root")
    p.add_argument("--output",      type=str, default="myModel.h5", help="Output .h5 filename")
    p.add_argument("--epochs",      type=int, default=20,           help="Total training epochs")
    p.add_argument("--fine_tune",   type=int, default=10,           help="Extra fine-tune epochs (unfrozen top layers)")
    p.add_argument("--lr",          type=float, default=1e-4,       help="Initial learning rate")
    p.add_argument("--auto_split",  type=bool, default=False,       help="Auto-split flat dataset into train/val")
    p.add_argument("--val_split",   type=float, default=0.2,        help="Validation split ratio (only if --auto_split)")
    return p.parse_args()


def build_generators(data_dir: str, auto_split: bool, val_split: float):
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
        validation_split=val_split if auto_split else 0.0,
    )
    val_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=val_split if auto_split else 0.0,
    )

    kwargs = dict(
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )

    if auto_split:
        train_gen = train_aug.flow_from_directory(data_dir, subset="training",   **kwargs)
        val_gen   = val_aug.flow_from_directory(data_dir,   subset="validation", **kwargs)
    else:
        train_gen = train_aug.flow_from_directory(os.path.join(data_dir, "train"), **kwargs)
        val_gen   = val_aug.flow_from_directory(os.path.join(data_dir, "val"),     **kwargs)

    return train_gen, val_gen


def build_model(num_classes: int) -> Model:
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # freeze base initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=out)
    return model, base


def train(args):
    print("\nLoading dataset...")
    train_gen, val_gen = build_generators(args.data_dir, args.auto_split, args.val_split)

    num_classes = train_gen.num_classes
    print(f"Found {num_classes} classes, {train_gen.samples} training images")

    # Save class index mapping (needed by FastAPI predict.py)
    import json
    class_map = {v: k for k, v in train_gen.class_indices.items()}
    with open("class_indices.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print("class_indices.json saved")

    print("\nBuilding model...")
    model, base = build_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint("best_model.h5", save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\nPhase 1: training top layers (base frozen)...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    print(f"\nPhase 2: fine-tuning top 30 layers for {args.fine_tune} more epochs...")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=args.lr / 10),  # lower LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.fine_tune,
        callbacks=callbacks,
    )

    model.save(args.output)
    print(f"\nModel saved to {args.output}")

    plot_history(history1, history2, args.output)


def plot_history(h1, h2, output_name):
    acc  = h1.history["accuracy"]  + h2.history["accuracy"]
    val  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss = h1.history["loss"] + h2.history["loss"]
    vloss= h1.history["val_loss"] + h2.history["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(acc,  label="Train Acc")
    axes[0].plot(val,  label="Val Acc")
    axes[0].axvline(len(h1.history["accuracy"]) - 1, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(loss,  label="Train Loss")
    axes[1].plot(vloss, label="Val Loss")
    axes[1].axvline(len(h1.history["loss"]) - 1, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].legend()

    plot_path = output_name.replace(".h5", "_training_curves.png")
    plt.savefig(plot_path)
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    args = parse_args()
    train(args)

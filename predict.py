"""
predict.py — Prediction logic used by FastAPI
----------------------------------------------
Handles image preprocessing, model inference, and CSV lookup.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = 224
TOP_K       = 3           # return top-3 predictions
MODEL_PATH  = "myModel.h5"
CSV_PATH    = "p4.csv"
CLASS_JSON  = "class_indices.json"


# ── Loader (called once at startup) ───────────────────────────────────────────
def load_artifacts():
    """
    Returns (model, class_map, disease_df).
    Call this once when FastAPI starts — not on every request.
    """
    import tensorflow as tf

    print(f" Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f" Loading class map from {CLASS_JSON} …")
    with open(CLASS_JSON, "r") as f:
        class_map = json.load(f)   # {str(index): "ClassName"}
    # Ensure keys are ints
    class_map = {int(k): v for k, v in class_map.items()}

    print(f" Loading disease CSV from {CSV_PATH} …")
    # p4.csv uses a pipe separator to avoid parsing issues with commas inside fields.
    df = pd.read_csv(CSV_PATH, sep="|")
    # Normalise column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    print("Artifacts loaded.")
    return model, class_map, df


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Raw bytes → normalised (1, 224, 224, 3) float32 array.
    Accepts JPEG, PNG, WebP — whatever PIL can open.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict(
    image_bytes: bytes,
    model,
    class_map: dict,
    disease_df: pd.DataFrame,
) -> dict:
    """
    Returns structured prediction dict with top-K results.

    Example response shape:
    {
      "top_predictions": [
        {
          "rank": 1,
          "class_name": "Tomato___Late_blight",
          "confidence": 0.92,
          "label": "Tomato Late Blight",
          "description": "...",
          "prevention": "...",
          "treatment": "...",
          "example_picture": "..."
        },
        ...
      ]
    }
    """
    arr = preprocess(image_bytes)
    probs = model.predict(arr, verbose=0)[0]           # shape: (num_classes,)
    # Ensure JSON-safe finite values (avoid NaN/inf breaking JSONResponse).
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    top_indices = np.argsort(probs)[::-1][:TOP_K]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        class_name  = class_map.get(idx, f"Unknown_{idx}")
        confidence  = float(probs[idx])

        # Look up disease info from CSV
        disease_info = lookup_disease(class_name, disease_df)

        results.append({
            "rank":            rank,
            "class_name":      class_name,
            "confidence":      round(confidence, 4),
            "confidence_pct":  f"{confidence * 100:.1f}%",
            **disease_info,
        })

    return {"top_predictions": results}


# ── CSV lookup ─────────────────────────────────────────────────────────────────
def lookup_disease(class_name: str, df: pd.DataFrame) -> dict:
    """
    Match class_name against the CSV's label column.
    Falls back gracefully if no match found.
    Expected CSV columns (after normalising):
        label, description, prevention, treatment, example_picture
    """
    # Try exact match first, then partial
    mask = df["label"].str.lower() == class_name.lower()
    if not mask.any():
        mask = df["label"].str.lower().str.contains(
            class_name.lower().replace("___", " ").replace("_", " "),
            na=False,
        )

    if mask.any():
        row = df[mask].iloc[0]
        # pandas can return NaN floats for missing CSV fields; JSON can't serialize NaN.
        label_val = row.get("label", class_name)
        description_val = row.get("description", "N/A")
        prevention_val = row.get("prevention", "N/A")
        treatment_val = row.get("treatment", "N/A")
        example_picture_val = row.get("example_picture", "")

        label_val = class_name if pd.isna(label_val) else label_val
        description_val = "N/A" if pd.isna(description_val) else description_val
        prevention_val = "N/A" if pd.isna(prevention_val) else prevention_val
        treatment_val = "N/A" if pd.isna(treatment_val) else treatment_val
        example_picture_val = "" if pd.isna(example_picture_val) else example_picture_val

        return {
            "label":            label_val,
            "description":      description_val,
            "prevention":       prevention_val,
            "treatment":        treatment_val,
            "example_picture":  example_picture_val,
        }

    # No match — return defaults
    return {
        "label":           class_name,
        "description":     "No information available.",
        "prevention":      "Consult an agricultural expert.",
        "treatment":       "Consult an agricultural expert.",
        "example_picture": "",
    }

# Plant Disease Detection

Plant Disease Detection is a machine learning web application that classifies plant leaf diseases from an uploaded image and returns actionable guidance.

## Features

- Upload a leaf image from the browser UI
- Run model inference through a FastAPI backend
- Return top-3 predictions with confidence scores
- Show disease description, prevention tips, and treatment suggestions from `p4.csv`

## Tech Stack

- Python
- FastAPI + Uvicorn
- TensorFlow / Keras (MobileNetV2)
- NumPy, Pandas, Pillow
- HTML/CSS/JavaScript frontend (`index.html`)

## Project Structure

- `main.py` - FastAPI app and routes
- `predict.py` - model loading, preprocessing, inference, CSV lookup
- `train.py` - model training/fine-tuning pipeline
- `index.html` - frontend UI
- `p4.csv` - disease metadata (pipe-separated)
- `requirements.txt` - Python dependencies

## Prerequisites

- Python 3.10+
- A trained model file in project root (for inference), e.g. `myModel.h5`
- `class_indices.json` in project root

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the API + Frontend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open in browser:

- `http://127.0.0.1:8000/` - frontend page
- `http://127.0.0.1:8000/health` - API health check
- `http://127.0.0.1:8000/classes` - model classes

## Prediction Endpoint

`POST /predict`

- Content type: `multipart/form-data`
- Field name: `file`
- Accepted image types: JPEG, PNG, WebP

Response includes:

- `status`
- `inference_time`
- `filename`
- `top_predictions` (rank, class_name, confidence, confidence_pct, label, description, prevention, treatment, example_picture)

## Notes

- `p4.csv` uses pipe (`|`) as separator to avoid comma parsing issues inside text fields.
- This project is intended for educational/research use and should not replace expert agronomic advice.

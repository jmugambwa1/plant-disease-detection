"""
main.py — FastAPI Plant Disease Detection API
---------------------------------------------
Endpoints:
    GET  /               → health check
    GET  /classes        → list all detectable diseases
    POST /predict        → upload leaf image → get prediction

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Run in production:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
import uvicorn
import time

from predict import load_artifacts, predict


# ── Globals (loaded once at startup) ──────────────────────────────────────────
MODEL      = None
CLASS_MAP  = None
DISEASE_DF = None
INDEX_PATH = Path(__file__).parent / "index.html"


# ── App lifecycle ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, CLASS_MAP, DISEASE_DF
    MODEL, CLASS_MAP, DISEASE_DF = load_artifacts()
    yield
    # Cleanup on shutdown (nothing needed)


app = FastAPI(
    title=" Plant Disease Detection API",
    description="Upload a leaf image → get AI-powered disease diagnosis and treatment.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Flutter app (any origin during dev — tighten in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Frontend"])
async def root():
    """Serve the frontend page."""
    if not INDEX_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_PATH)


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint for API monitoring."""
    return {
        "status":  "ok",
        "message": "Plant Disease Detection API is running ",
        "model":   "MobileNetV2 fine-tuned on PlantVillage",
    }


@app.get("/classes", tags=["Info"])
async def get_classes():
    """Return all detectable disease classes."""
    if CLASS_MAP is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {
        "total":   len(CLASS_MAP),
        "classes": list(CLASS_MAP.values()),
    }


@app.post("/predict", tags=["Prediction"])
async def predict_disease(file: UploadFile = File(...)):
    """
    Upload a plant leaf image (JPEG/PNG/WebP).
    Returns top-3 predictions with disease info.
    """
    # Validate file type
    ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if file.content_type not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Upload JPEG, PNG, or WebP.",
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(image_bytes) > 10 * 1024 * 1024:   # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    # Run inference
    start = time.perf_counter()
    try:
        result = predict(image_bytes, MODEL, CLASS_MAP, DISEASE_DF)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    elapsed = round(time.perf_counter() - start, 3)

    return JSONResponse(content={
        "status":          "success",
        "inference_time":  f"{elapsed}s",
        "filename":        file.filename,
        **result,
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

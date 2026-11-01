"""
server.py
=========
FastAPI backend for Traffic Sign Classification.
Serves predictions using the trained Keras model (`models/traffic_sign_cnn.keras`).

Deployable on Render, Railway, or run locally:
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import io
import json
from typing import List, Optional
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sign_classes import CLASS_NAMES

IMG_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "traffic_sign_cnn.keras")
EVAL_PATH = os.path.join(os.path.dirname(__file__), "models", "evaluation.json")

app = FastAPI(
    title="Traffic Sign Classification API",
    description="REST API for predicting traffic signs using a CNN trained on the GTSRB dataset.",
    version="1.0.0",
)

# Enable CORS for frontend (e.g. Vercel deployment and local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, can be restricted to specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None

def get_model():
    """Lazy-load the trained Keras model so the API starts cleanly."""
    global _model
    if _model is None:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at '{MODEL_PATH}'. Run 'python train.py' to generate it."
            )
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


class PredictionItem(BaseModel):
    classId: int
    name: str
    confidence: float


class PredictResponse(BaseModel):
    classId: int
    name: str
    confidence: float
    top3: List[PredictionItem]
    modelSource: str = "Python TensorFlow (GTSRB trained)"


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Traffic Sign Classification API",
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health",
            "evaluation": "GET /api/evaluation",
        },
    }


@app.get("/api/health")
def health_check():
    model_loaded = os.path.isfile(MODEL_PATH)
    return {
        "status": "healthy",
        "modelAvailable": model_loaded,
        "classesCount": len(CLASS_NAMES),
        "dataset": "GTSRB (German Traffic Sign Recognition Benchmark)",
    }


@app.get("/api/evaluation")
def get_evaluation():
    """Returns official evaluation metrics from the Python training run if available."""
    if os.path.isfile(EVAL_PATH):
        try:
            with open(EVAL_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read evaluation data: {str(e)}")
    return {
        "status": "pending",
        "message": "Evaluation metrics not yet generated. Run 'python train.py' to train and evaluate.",
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_traffic_sign(file: UploadFile = File(...)):
    """Classify an uploaded traffic sign image using the trained CNN."""
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"] and file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.",
        )

    # Read and decode image
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file or decoding failed.")

    # Preprocessing: resize to 32x32 and normalize pixels to [0, 1]
    resized_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(resized_img, dtype=np.float32) / 255.0  # (32, 32, 3)
    input_tensor = np.expand_dims(arr, axis=0)              # (1, 32, 32, 3)

    # Run inference
    try:
        model = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

    probs = model.predict(input_tensor, verbose=0)[0]  # Softmax probabilities over 43 classes
    top_indices = np.argsort(probs)[::-1][:3]

    top3 = [
        PredictionItem(
            classId=int(idx),
            name=CLASS_NAMES[int(idx)],
            confidence=float(probs[idx]),
        )
        for idx in top_indices
    ]
    best = top3[0]

    return PredictResponse(
        classId=best.classId,
        name=best.name,
        confidence=best.confidence,
        top3=top3,
        modelSource="Python TensorFlow (GTSRB trained)",
    )

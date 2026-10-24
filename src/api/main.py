from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.train.dataset import FASHION_MNIST_CLASSES, preprocess_image
from . import model_loader
from .schemas import HealthResponse, PredictionResponse

logger = logging.getLogger(__name__)

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp", "image/gif", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_loader.load_model()
        logger.info("Model loaded successfully (run_id=%s)", model_loader.get_run_id())
    except Exception as e:
        logger.warning("Model failed to load at startup: %s", e)
        logger.warning("POST /predict will return 503 until the model is available.")
    yield


app = FastAPI(
    title="Fashion-MNIST Inference API",
    description="Upload a 28×28 grayscale image and get a Fashion-MNIST class prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
        model_run_id=model_loader.get_run_id(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check MODEL_RUN_ID and MinIO connectivity.",
        )

    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{content_type}'. Accepted: {sorted(_ALLOWED_CONTENT_TYPES)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        img = Image.open(io.BytesIO(data))  # re-open after verify (verify closes the file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid or corrupt image: {e}") from e

    tensor = preprocess_image(img).to(model_loader.get_device())
    model = model_loader.get_model()

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    class_idx = int(probs.argmax())
    probabilities = {name: float(probs[i]) for i, name in enumerate(FASHION_MNIST_CLASSES)}

    return PredictionResponse(
        predicted_class=FASHION_MNIST_CLASSES[class_idx],
        class_index=class_idx,
        confidence=float(probs[class_idx]),
        probabilities=probabilities,
    )

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch

from src.train.config import TrainingConfig
from src.train.model import LeNet5
from src.train.storage import download_model

_model: LeNet5 | None = None
_run_id: str | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model() -> None:
    """Download model from MinIO and load into memory. Called at API startup."""
    global _model, _run_id

    run_id = os.getenv("MODEL_RUN_ID", "").strip()
    if not run_id:
        raise RuntimeError(
            "MODEL_RUN_ID environment variable is not set. "
            "Run training first and set MODEL_RUN_ID to the MLflow run ID."
        )

    cfg = TrainingConfig()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        download_model(cfg, run_id, tmp_path)
        model = LeNet5()
        model.load_state_dict(torch.load(tmp_path, map_location=_device))
        model.to(_device)
        model.eval()
    finally:
        tmp_path.unlink(missing_ok=True)

    _model = model
    _run_id = run_id


def get_model() -> LeNet5:
    if _model is None:
        raise RuntimeError("Model is not loaded. Check MODEL_RUN_ID and MinIO connectivity.")
    return _model


def is_loaded() -> bool:
    return _model is not None


def get_run_id() -> str | None:
    return _run_id


def get_device() -> torch.device:
    return _device

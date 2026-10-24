from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from fastapi.testclient import TestClient

from src.train.model import LeNet5


@pytest.fixture
def sample_image_bytes() -> bytes:
    """28x28 grayscale PNG in memory."""
    img = Image.new("L", (28, 28), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_rgb_image_bytes() -> bytes:
    """28x28 RGB PNG in memory."""
    img = Image.new("RGB", (28, 28), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def loaded_model() -> LeNet5:
    model = LeNet5()
    model.eval()
    return model


@pytest.fixture
def api_client(loaded_model):
    """FastAPI TestClient with a pre-loaded model (no MinIO needed)."""
    import src.api.model_loader as loader
    from src.api.main import app

    with patch.object(loader, "_model", loaded_model), \
         patch.object(loader, "_run_id", "test-run-001"):
        with TestClient(app, raise_server_exceptions=True) as client:
            yield client

from __future__ import annotations

import pytest
import torch

from src.train.model import LeNet5
from src.train.dataset import FASHION_MNIST_CLASSES


def test_model_output_shape():
    model = LeNet5()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 10)


def test_model_batch_output_shape():
    model = LeNet5()
    model.eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 10)


def test_softmax_sums_to_one():
    model = LeNet5()
    model.eval()
    x = torch.randn(1, 1, 28, 28)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_class_names_count():
    assert len(FASHION_MNIST_CLASSES) == 10


def test_prediction_response_via_api(api_client, sample_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in FASHION_MNIST_CLASSES
    assert body["class_index"] in range(10)
    probs = body["probabilities"]
    assert len(probs) == 10
    assert abs(sum(probs.values()) - 1.0) < 1e-4

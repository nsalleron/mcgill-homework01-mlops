from __future__ import annotations

import io

import pytest
from PIL import Image


def test_health_returns_200(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_run_id"] == "test-run-001"


def test_predict_valid_image(api_client, sample_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.png", sample_image_bytes, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "predicted_class" in body
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rgb_image(api_client, sample_rgb_image_bytes):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.png", sample_rgb_image_bytes, "image/png")},
    )
    assert resp.status_code == 200


def test_predict_unsupported_type(api_client):
    resp = api_client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_corrupt_image(api_client):
    resp = api_client.post(
        "/predict",
        files={"file": ("bad.png", b"\x89PNG corrupt", "image/png")},
    )
    assert resp.status_code == 422


def test_predict_empty_file(api_client):
    resp = api_client.post(
        "/predict",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 422

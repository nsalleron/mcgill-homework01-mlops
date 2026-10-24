from __future__ import annotations

import io

import pytest
import torch
from PIL import Image

from src.train.dataset import preprocess_image


def _make_image(mode: str, size: tuple[int, int] = (100, 100)) -> Image.Image:
    return Image.new(mode, size, color=128)


def test_grayscale_output_shape():
    img = _make_image("L")
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 1, 28, 28)


def test_rgb_converted_to_grayscale():
    img = _make_image("RGB")
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 1, 28, 28)


def test_output_dtype():
    img = _make_image("L")
    tensor = preprocess_image(img)
    assert tensor.dtype == torch.float32


def test_normalization_range():
    img = _make_image("L", (28, 28))
    tensor = preprocess_image(img)
    # Normalized with mean=0.5, std=0.5: range roughly [-1, 1]
    assert tensor.min() >= -2.0
    assert tensor.max() <= 2.0


def test_large_image_resized():
    img = _make_image("RGB", (512, 512))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 1, 28, 28)

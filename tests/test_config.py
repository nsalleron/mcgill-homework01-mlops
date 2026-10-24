from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.train.config import TrainingConfig


def test_default_config_loads():
    cfg = TrainingConfig()
    assert cfg.epochs > 0
    assert 0 < cfg.val_split < 1
    assert cfg.batch_size > 0
    assert cfg.learning_rate > 0


def test_optimizer_valid_values():
    cfg = TrainingConfig(optimizer="sgd")
    assert cfg.optimizer == "sgd"
    cfg = TrainingConfig(optimizer="adam")
    assert cfg.optimizer == "adam"


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("EPOCHS", "5")
    monkeypatch.setenv("BATCH_SIZE", "32")
    cfg = TrainingConfig()
    assert cfg.epochs == 5
    assert cfg.batch_size == 32


def test_minio_defaults():
    cfg = TrainingConfig()
    assert cfg.minio_bucket == "fashion-minst-bucket"
    assert cfg.minio_prefix == "model"


def test_invalid_optimizer_raises():
    with pytest.raises(Exception):
        TrainingConfig(optimizer="invalid_optimizer")

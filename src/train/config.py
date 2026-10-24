from __future__ import annotations

import toml
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_TOML_PATH = Path(__file__).parent.parent.parent / "config" / "training.toml"


def _load_toml_defaults() -> dict:
    if _TOML_PATH.exists():
        data = toml.load(_TOML_PATH)
        t = data.get("training", {})
        op = t.get("optimizer_params", {})
        sched = t.get("scheduler", {})
        es = t.get("early_stopping", {})
        dl = t.get("dataloader", {})
        ml = data.get("mlflow", {})
        mn = data.get("minio", {})
        return {
            "epochs": t.get("epochs", 10),
            "batch_size": t.get("batch_size", 64),
            "learning_rate": t.get("learning_rate", 0.001),
            "optimizer": t.get("optimizer", "adam"),
            "loss_fn": t.get("loss_fn", "cross_entropy"),
            "seed": t.get("seed", 42),
            "val_split": t.get("val_split", 0.1),
            "weight_decay": op.get("weight_decay", 1e-4),
            "momentum": op.get("momentum", 0.9),
            "scheduler_type": sched.get("type", ""),
            "scheduler_step_size": sched.get("step_size", 5),
            "scheduler_gamma": sched.get("gamma", 0.5),
            "early_stopping_enabled": es.get("enabled", False),
            "early_stopping_patience": es.get("patience", 3),
            "num_workers": dl.get("num_workers", 2),
            "pin_memory": dl.get("pin_memory", True),
            "mlflow_experiment_name": ml.get("experiment_name", "fashion-mnist-lenet5"),
            "minio_bucket": mn.get("bucket", "fashion-minst-bucket"),
            "minio_prefix": mn.get("prefix", "model"),
        }
    return {}


_DEFAULTS = _load_toml_defaults()


class TrainingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Core training
    epochs: int = Field(default=_DEFAULTS.get("epochs", 10))
    batch_size: int = Field(default=_DEFAULTS.get("batch_size", 64))
    learning_rate: float = Field(default=_DEFAULTS.get("learning_rate", 0.001))
    optimizer: Literal["adam", "adamw", "sgd"] = Field(default=_DEFAULTS.get("optimizer", "adam"))
    loss_fn: Literal["cross_entropy", "nll_loss"] = Field(default=_DEFAULTS.get("loss_fn", "cross_entropy"))
    seed: int = Field(default=_DEFAULTS.get("seed", 42))
    val_split: float = Field(default=_DEFAULTS.get("val_split", 0.1))

    # Optimizer extras
    weight_decay: float = Field(default=_DEFAULTS.get("weight_decay", 1e-4))
    momentum: float = Field(default=_DEFAULTS.get("momentum", 0.9))

    # Scheduler
    scheduler_type: str = Field(default=_DEFAULTS.get("scheduler_type", ""))
    scheduler_step_size: int = Field(default=_DEFAULTS.get("scheduler_step_size", 5))
    scheduler_gamma: float = Field(default=_DEFAULTS.get("scheduler_gamma", 0.5))

    # Early stopping
    early_stopping_enabled: bool = Field(default=_DEFAULTS.get("early_stopping_enabled", False))
    early_stopping_patience: int = Field(default=_DEFAULTS.get("early_stopping_patience", 3))

    # DataLoader
    num_workers: int = Field(default=_DEFAULTS.get("num_workers", 2))
    pin_memory: bool = Field(default=_DEFAULTS.get("pin_memory", True))

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")
    mlflow_experiment_name: str = Field(default=_DEFAULTS.get("mlflow_experiment_name", "fashion-mnist-lenet5"))

    # MinIO
    minio_endpoint: str = Field(default="localhost:9000")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin")
    minio_secure: bool = Field(default=False)
    minio_bucket: str = Field(default=_DEFAULTS.get("minio_bucket", "fashion-minst-bucket"))
    minio_prefix: str = Field(default=_DEFAULTS.get("minio_prefix", "model"))

    # Webhook
    webhook_url: str = Field(default="")

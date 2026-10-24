from __future__ import annotations

import random
import tempfile
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .model import LeNet5


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> torch.optim.Optimizer:
    params = model.parameters()
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    return torch.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.momentum, weight_decay=cfg.weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig):
    if cfg.scheduler_type == "step":
        return StepLR(optimizer, step_size=cfg.scheduler_step_size, gamma=cfg.scheduler_gamma)
    if cfg.scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    return None


def _build_loss(cfg: TrainingConfig) -> nn.Module:
    if cfg.loss_fn == "nll_loss":
        return nn.NLLLoss()
    return nn.CrossEntropyLoss()


def train(
    cfg: TrainingConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
) -> tuple[str, float, "LeNet5"]:
    """Train the model and log to MLflow. Returns (run_id, val_accuracy, trained_model)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = LeNet5().to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)
    criterion = _build_loss(cfg)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    with mlflow.start_run() as run:
        mlflow.log_params({
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "optimizer": cfg.optimizer,
            "loss_fn": cfg.loss_fn,
            "seed": cfg.seed,
            "val_split": cfg.val_split,
            "weight_decay": cfg.weight_decay,
            "momentum": cfg.momentum,
            "scheduler_type": cfg.scheduler_type or "none",
            "early_stopping": cfg.early_stopping_enabled,
            "early_stopping_patience": cfg.early_stopping_patience,
        })

        best_val_loss = float("inf")
        patience_counter = 0
        val_accuracy = 0.0

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * images.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    correct += (outputs.argmax(1) == labels).sum().item()
            val_loss /= len(val_loader.dataset)
            val_accuracy = correct / len(val_loader.dataset)

            mlflow.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy},
                step=epoch,
            )
            print(f"Epoch {epoch}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_accuracy:.4f}")

            if scheduler:
                scheduler.step()

            if cfg.early_stopping_enabled:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.early_stopping_patience:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        mlflow.log_metric("final_val_accuracy", val_accuracy)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        run_id = run.info.run_id

    print(f"MLflow run_id: {run_id}  final_val_accuracy: {val_accuracy:.4f}")
    return run_id, val_accuracy, model

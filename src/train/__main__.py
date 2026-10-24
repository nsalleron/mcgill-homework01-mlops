from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import torch

from .config import TrainingConfig
from .dataset import get_dataloaders
from .storage import upload_model
from .trainer import train
from .webhook import send_webhook

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    cfg = TrainingConfig()
    logger.info("Training config loaded")

    train_loader, val_loader, _ = get_dataloaders(
        val_split=cfg.val_split,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    run_id, val_accuracy, model = train(cfg, train_loader, val_loader)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_file = Path(f.name)

    try:
        torch.save(model.state_dict(), model_file)
        object_key = upload_model(cfg, run_id, model_file)
    finally:
        model_file.unlink(missing_ok=True)

    send_webhook(cfg.webhook_url, run_id, cfg.minio_bucket, object_key, val_accuracy)

    print(f"\nTraining complete.")
    print(f"  MLflow run_id : {run_id}")
    print(f"  Val accuracy  : {val_accuracy:.4f}")
    print(f"  MinIO object  : s3://{cfg.minio_bucket}/{object_key}")
    print(f"\nSet MODEL_RUN_ID={run_id} in .env before starting the API.")


if __name__ == "__main__":
    main()

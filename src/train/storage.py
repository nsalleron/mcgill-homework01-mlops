from __future__ import annotations

from pathlib import Path

from minio import Minio
from minio.error import S3Error

from .config import TrainingConfig


def get_client(cfg: TrainingConfig) -> Minio:
    return Minio(
        cfg.minio_endpoint,
        access_key=cfg.minio_access_key,
        secret_key=cfg.minio_secret_key,
        secure=cfg.minio_secure,
    )


def upload_model(cfg: TrainingConfig, run_id: str, model_path: Path) -> str:
    """Upload model.pt to MinIO and return the object key."""
    client = get_client(cfg)

    if not client.bucket_exists(cfg.minio_bucket):
        client.make_bucket(cfg.minio_bucket)
        print(f"Created bucket: {cfg.minio_bucket}")

    object_key = f"{cfg.minio_prefix}/{run_id}/model.pt"
    client.fput_object(
        cfg.minio_bucket,
        object_key,
        str(model_path),
        content_type="application/octet-stream",
    )
    print(f"Uploaded model to s3://{cfg.minio_bucket}/{object_key}")
    return object_key


def download_model(cfg: TrainingConfig, run_id: str, dest_path: Path) -> None:
    """Download model.pt from MinIO to dest_path."""
    client = get_client(cfg)
    object_key = f"{cfg.minio_prefix}/{run_id}/model.pt"
    try:
        client.fget_object(cfg.minio_bucket, object_key, str(dest_path))
    except S3Error as e:
        raise FileNotFoundError(
            f"Model not found in MinIO at s3://{cfg.minio_bucket}/{object_key}: {e}"
        ) from e

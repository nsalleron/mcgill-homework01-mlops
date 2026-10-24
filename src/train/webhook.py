from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


def send_webhook(
    url: str,
    run_id: str,
    bucket: str,
    object_key: str,
    val_accuracy: float,
) -> None:
    """POST a training-complete notification. Non-blocking — logs errors, does not raise."""
    if not url:
        logger.info("WEBHOOK_URL not configured, skipping notification")
        return

    payload = {
        "status": "success",
        "mlflow_run_id": run_id,
        "model_bucket": bucket,
        "model_object_key": object_key,
        "val_accuracy": round(val_accuracy, 6),
    }
    try:
        resp = httpx.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Webhook sent to %s → HTTP %s", url, resp.status_code)
    except httpx.HTTPError as e:
        logger.error("Webhook delivery failed: %s", e)

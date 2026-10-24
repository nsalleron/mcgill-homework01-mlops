# Fashion-MNIST MLOps

End-to-end MLOps pipeline: train a LeNet-5 CNN on Fashion-MNIST, track runs in MLflow, store the model in MinIO, and serve predictions via FastAPI + Streamlit.

> **Note:** The MinIO bucket name `fashion-minst-bucket` is intentional and matches the assignment specification.

---

## Architecture

```
Train (LeNet-5 on Fashion-MNIST)
  → Log params/metrics/model to MLflow
  → Upload model.pt to MinIO (fashion-minst-bucket/model/<run_id>/model.pt)
  → POST training-complete webhook

FastAPI (loads model from MinIO at startup)
  ← Streamlit UI (upload image → predicted class)
```

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/)
- [uv](https://docs.astral.sh/uv/) (for local development / tests)
- Python 3.11+

---

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env if you need custom MinIO/MLflow endpoints or credentials
```

### 2. Start infrastructure (MinIO + MLflow)

```bash
docker compose up -d minio mlflow
```

- MinIO console: http://localhost:9001 (user: `minioadmin` / pass: `minioadmin`)
- MLflow UI: http://localhost:5000

### 3. Run training

```bash
docker compose --profile train run --rm train
```

Training output will print the `run_id` at the end:

```
Training complete.
  MLflow run_id : abc123def456
  Val accuracy  : 0.8912
  MinIO object  : s3://fashion-minst-bucket/model/abc123def456/model.pt

Set MODEL_RUN_ID=abc123def456 in .env before starting the API.
```

Add that run ID to `.env`:

```bash
echo "MODEL_RUN_ID=abc123def456" >> .env
```

### 4. Start the prediction stack

```bash
docker compose up -d api ui
```

- API docs: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501

---

## Configuration

All training settings live in `config/training.toml`. They can be overridden per-run via environment variables:

| TOML key | Env var | Default | Description |
|---|---|---|---|
| `training.epochs` | `EPOCHS` | `10` | Number of training epochs |
| `training.batch_size` | `BATCH_SIZE` | `64` | Mini-batch size |
| `training.learning_rate` | `LEARNING_RATE` | `0.001` | Initial learning rate |
| `training.optimizer` | `OPTIMIZER` | `adam` | `adam` / `adamw` / `sgd` |
| `training.loss_fn` | `LOSS_FN` | `cross_entropy` | `cross_entropy` / `nll_loss` |
| `training.seed` | `SEED` | `42` | Random seed for reproducibility |
| `training.val_split` | `VAL_SPLIT` | `0.1` | Fraction of training data for validation |
| `training.early_stopping.enabled` | `EARLY_STOPPING_ENABLED` | `false` | Enable early stopping |
| `training.early_stopping.patience` | `EARLY_STOPPING_PATIENCE` | `3` | Epochs without improvement before stopping |
| `training.scheduler.type` | `SCHEDULER_TYPE` | `""` | `step` / `cosine` / empty (none) |

Example — run 20 epochs with SGD:

```bash
EPOCHS=20 OPTIMIZER=sgd docker compose --profile train run --rm train
```

---

## MLflow Experiment Tracking

Open http://localhost:5000 after starting MLflow.

- Experiment name: **fashion-mnist-lenet5** (configurable via `MLFLOW_EXPERIMENT_NAME`)
- Each run logs: all hyperparameters, train/val loss per epoch, final val accuracy
- Model artifact is stored in the MLflow artifact store and separately in MinIO

To compare runs, select multiple rows in the MLflow UI and click **Compare**.

---

## MinIO Model Storage

Open http://localhost:9001 after starting MinIO.

Models are stored under:
```
fashion-minst-bucket/model/<run_id>/model.pt
```

The bucket is created automatically at training time if it does not exist.

### Selecting which model the API loads

Set `MODEL_RUN_ID` in `.env` (or as an environment variable) to the desired MLflow `run_id`. The API downloads `model/<MODEL_RUN_ID>/model.pt` from MinIO at startup.

If `MODEL_RUN_ID` is empty, the API starts but `/predict` returns `503` until a model is available.

---

## Training-Complete Webhook

Set `WEBHOOK_URL` in `.env` to any HTTP endpoint. After a successful MinIO upload, the training script POSTs:

```json
{
  "status": "success",
  "mlflow_run_id": "abc123def456",
  "model_bucket": "fashion-minst-bucket",
  "model_object_key": "model/abc123def456/model.pt",
  "val_accuracy": 0.8912
}
```

**Testing locally with [webhook.site](https://webhook.site):**

1. Open https://webhook.site and copy your unique URL
2. Set `WEBHOOK_URL=https://webhook.site/YOUR-TOKEN` in `.env`
3. Run training — the payload appears in the webhook.site dashboard

The webhook is non-blocking: if the URL is unreachable, training output logs an error but does not fail the process.

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_run_id": "abc123def456"
}
```

### `POST /predict`

**Request:** multipart form with field `file` (JPEG, PNG, BMP)

**Response:**

```json
{
  "predicted_class": "Sneaker",
  "class_index": 7,
  "confidence": 0.9423,
  "probabilities": {
    "T-shirt/top": 0.002,
    "Trouser": 0.001,
    ...
    "Sneaker": 0.9423,
    ...
  }
}
```

Images are automatically resized to 28×28 and converted to grayscale.

---

## Local Development

```bash
# Install all dependencies
uv sync --all-groups

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src --cov-report=term-missing

# Run API locally (requires MinIO running and MODEL_RUN_ID set)
uv run uvicorn src.api.main:app --reload

# Run Streamlit locally (requires API running)
uv run streamlit run src/ui/app.py
```

---

## Project Structure

```
.
├── config/training.toml     # Default hyperparameters
├── docker/                  # Dockerfiles (train / api / ui)
├── docker-compose.yml       # Full stack (minio, mlflow, api, ui, train)
├── src/
│   ├── train/               # Training pipeline (model, dataset, trainer, storage, webhook)
│   ├── api/                 # FastAPI inference service
│   └── ui/                  # Streamlit frontend
└── tests/                   # pytest tests
```

---

## Startup Sequence Summary

```bash
# Infrastructure
docker compose up -d minio mlflow

# Training (one-shot — wait for it to finish)
docker compose --profile train run --rm train
# → note the printed MODEL_RUN_ID, add to .env

# Prediction stack
docker compose up -d api ui
# → open http://localhost:8501
```

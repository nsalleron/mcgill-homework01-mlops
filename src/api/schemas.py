from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_run_id: str | None = None

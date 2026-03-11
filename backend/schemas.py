"""
Pydantic v2 schemas for all API request / response models.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ══════════════════════════════════════════════════════
# Request schemas
# ══════════════════════════════════════════════════════


class TextPredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw source-code text to analyse")
    model_name: Optional[str] = Field(
        "WebshellDetector",
        description="Model: WebshellDetector | TextCNN | TextRNN | TransformerClassifier | ensemble",
    )


# ══════════════════════════════════════════════════════
# Sub-schemas
# ══════════════════════════════════════════════════════


class FileFeatureSchema(BaseModel):
    """Heuristic features extracted from file content (for explainability)."""

    filename: str
    file_size_bytes: int
    extension: str
    line_count: int
    entropy: float = Field(..., description="Shannon entropy (bits/char)")
    entropy_risk: str = Field(..., description="Low | Medium | High")
    dangerous_func_count: int
    dangerous_funcs_found: list[str]
    obfuscation_score: int
    heuristic_risk: float = Field(..., description="Composite risk score 0–100")
    risk_label: str = Field(..., description="Low | Medium | High")


class SingleModelVote(BaseModel):
    """One model's vote inside an ensemble result."""

    model: str
    is_webshell: bool
    raw_score: float
    confidence: float


# ══════════════════════════════════════════════════════
# Core prediction result
# ══════════════════════════════════════════════════════


class PredictionResult(BaseModel):
    model: str
    is_webshell: bool
    confidence: float = Field(..., description="Confidence in the predicted class (0–1)")
    raw_score: float = Field(..., description="Raw sigmoid output (0–1)")
    label: str = Field(..., description="'Webshell Detected' or 'Normal Content'")
    explanation: Optional[str] = Field(None, description="Natural language reasoning for the prediction")
    # Optional fields (populated when available)

    features: Optional[FileFeatureSchema] = None
    votes: Optional[list[SingleModelVote]] = None  # ensemble sub-results


class BulkPredictionResult(BaseModel):
    filename: str
    result: PredictionResult


# ══════════════════════════════════════════════════════
# Utility responses
# ══════════════════════════════════════════════════════


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    available_models: list[str]
    environment: str
    message: str


class ModelStatsResponse(BaseModel):
    """Known benchmark metrics for each model (from dissertation experiments)."""

    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    params: str  # human-readable param count

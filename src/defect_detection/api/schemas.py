"""Pydantic schemas for the Visual Defect Detection API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PredictionLiteral = Literal["normal", "defective"]


class PredictionOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    created_at: str | None = None
    filename: str | None = None
    prediction: PredictionLiteral
    defect_type: str = "none"
    confidence: float = Field(ge=0, le=1)
    defective_probability: float = Field(ge=0, le=1)
    probabilities: dict[str, float] = Field(default_factory=dict)
    model_name: str
    model_version: str
    image_size: int
    device: str
    latency_ms: float
    image_width: int = 0
    image_height: int = 0
    heatmap_png_base64: str | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    latency_ms: float
    results: list[PredictionOut]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    device: str | None = None
    database: Literal["ok", "unavailable"] = "ok"
    total_inspections: int = 0
    version: str
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    architecture: str
    trained_at: str | None = None
    classes: list[str]
    image_size: int
    threshold: float
    device: str
    model_path: str
    model_size_mb: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    epoch: int | None = None
    normalization: dict[str, list[float]] | None = None


class PredictionRecordOut(BaseModel):
    id: int
    created_at: str | None = None
    filename: str
    image_sha256: str = ""
    content_type: str = "image/jpeg"
    size_bytes: int = 0
    image_width: int = 0
    image_height: int = 0
    prediction: str
    defect_type: str
    confidence: float
    defective_probability: float
    model_name: str
    model_version: str
    latency_ms: float
    device: str
    status: str
    error: str | None = None


class MonitoringSummary(BaseModel):
    total_inspections: int
    window_size: int
    normal_count: int
    defective_count: int
    failure_count: int
    defect_rate: float | None
    defect_types: dict[str, int]
    avg_confidence: float | None
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    devices: dict[str, int]
    model_versions: dict[str, int]
    last_24h: int
    daily: list[dict[str, Any]]
    last_inspection_at: str | None


__all__ = [
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "MonitoringSummary",
    "PredictionOut",
    "PredictionRecordOut",
]

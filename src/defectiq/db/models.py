"""SQLAlchemy models for inspection history (Feature 8)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    """One inspected image and the model's decision."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )

    filename: Mapped[str] = mapped_column(String(255), default="upload")
    image_sha256: Mapped[str] = mapped_column(String(64), default="", index=True)
    content_type: Mapped[str] = mapped_column(String(100), default="image/jpeg")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    image_width: Mapped[int] = mapped_column(Integer, default=0)
    image_height: Mapped[int] = mapped_column(Integer, default=0)

    prediction: Mapped[str] = mapped_column(String(16), index=True)
    defect_type: Mapped[str] = mapped_column(String(64), default="none")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    defective_probability: Mapped[float] = mapped_column(Float, default=0.0)

    model_name: Mapped[str] = mapped_column(String(64), default="resnet18")
    model_version: Mapped[str] = mapped_column(String(80), default="")
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    device: Mapped[str] = mapped_column(String(16), default="cpu")

    status: Mapped[str] = mapped_column(String(16), default="success", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": isoformat(self.created_at),
            "filename": self.filename,
            "image_sha256": self.image_sha256,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "prediction": self.prediction,
            "defect_type": self.defect_type,
            "confidence": round(float(self.confidence), 4),
            "defective_probability": round(float(self.defective_probability), 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round(float(self.latency_ms), 2),
            "device": self.device,
            "status": self.status,
            "error": self.error,
        }


__all__ = ["Base", "PredictionRecord", "isoformat", "utcnow"]

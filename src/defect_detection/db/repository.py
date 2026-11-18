"""Repository for inspection records and monitoring aggregates (Feature 8).

SQLite by default, PostgreSQL via ``DEFECT_DATABASE_URL``. Every method returns
plain dictionaries; sessions are short lived and never leak into the API layer.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from defect_detection.config import Settings, get_settings
from defect_detection.db.models import Base, PredictionRecord, isoformat
from defect_detection.logging_config import get_logger

logger = get_logger(__name__)


class Repository:
    def __init__(
        self,
        database_url: str | None = None,
        *,
        settings: Settings | None = None,
        echo: bool = False,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        self.database_url = database_url or cfg.database_url

        if self.database_url.startswith("sqlite:///"):
            path = Path(self.database_url[len("sqlite:///") :])
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)

        connect_args = (
            {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        )
        self.engine = create_engine(
            self.database_url, echo=echo, future=True, connect_args=connect_args
        )
        self._session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def log_prediction(
        self,
        *,
        filename: str,
        image_sha256: str = "",
        content_type: str = "image/jpeg",
        size_bytes: int = 0,
        image_width: int = 0,
        image_height: int = 0,
        prediction: str,
        defect_type: str,
        confidence: float,
        defective_probability: float,
        model_name: str,
        model_version: str,
        latency_ms: float,
        device: str = "cpu",
        status: str = "success",
        error: str | None = None,
    ) -> dict[str, Any]:
        record = PredictionRecord(
            filename=filename[:255],
            image_sha256=image_sha256,
            content_type=content_type,
            size_bytes=size_bytes,
            image_width=image_width,
            image_height=image_height,
            prediction=prediction,
            defect_type=defect_type,
            confidence=float(confidence),
            defective_probability=float(defective_probability),
            model_name=model_name,
            model_version=model_version,
            latency_ms=float(latency_ms),
            device=device,
            status=status,
            error=error,
        )
        with self._session_factory() as session:
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.to_dict()

    def clear(self) -> None:
        with self._session_factory() as session:
            session.query(PredictionRecord).delete()
            session.commit()

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def recent(
        self,
        *,
        limit: int = 50,
        prediction: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            query = select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)
            if prediction:
                query = query.where(PredictionRecord.prediction == prediction)
            if status:
                query = query.where(PredictionRecord.status == status)
            return [record.to_dict() for record in session.scalars(query).all()]

    def count(self) -> int:
        with self._session_factory() as session:
            return int(session.scalar(select(func.count(PredictionRecord.id))) or 0)

    def summary(self, *, recent_limit: int = 1000) -> dict[str, Any]:
        with self._session_factory() as session:
            records = session.scalars(
                select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(recent_limit)
            ).all()
            total = int(session.scalar(select(func.count(PredictionRecord.id))) or 0)

        if not records:
            return {
                "total_inspections": 0,
                "window_size": 0,
                "normal_count": 0,
                "defective_count": 0,
                "failure_count": 0,
                "defect_rate": None,
                "defect_types": {},
                "avg_confidence": None,
                "avg_latency_ms": None,
                "p95_latency_ms": None,
                "devices": {},
                "model_versions": {},
                "last_24h": 0,
                "daily": [],
                "last_inspection_at": None,
            }

        successes = [row for row in records if row.status == "success"]
        defective = sum(1 for row in successes if row.prediction == "defective")
        latencies = np.asarray([row.latency_ms for row in records], dtype=float)
        confidences = np.asarray([row.confidence for row in successes], dtype=float)
        now = datetime.now(timezone.utc)

        def _utc(value: datetime) -> datetime:
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

        daily: dict[str, list[float]] = {}
        for row in records:
            key = _utc(row.created_at).date().isoformat()
            daily.setdefault(key, []).append(row.latency_ms)

        return {
            "total_inspections": total,
            "window_size": len(records),
            "normal_count": sum(1 for row in successes if row.prediction == "normal"),
            "defective_count": defective,
            "failure_count": sum(1 for row in records if row.status == "failed"),
            "defect_rate": round(defective / len(successes), 4) if successes else None,
            "defect_types": dict(
                Counter(row.defect_type for row in successes if row.defect_type != "none")
            ),
            "avg_confidence": round(float(confidences.mean()), 4) if len(confidences) else None,
            "avg_latency_ms": round(float(latencies.mean()), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "devices": dict(Counter(row.device for row in records)),
            "model_versions": dict(Counter(row.model_version for row in records)),
            "last_24h": int(
                sum(1 for row in records if _utc(row.created_at) >= now - timedelta(hours=24))
            ),
            "daily": [
                {
                    "date": key,
                    "inspections": len(values),
                    "avg_latency_ms": round(float(np.mean(values)), 2),
                }
                for key, values in sorted(daily.items())
            ],
            "last_inspection_at": isoformat(max(_utc(row.created_at) for row in records)),
        }

    def close(self) -> None:
        self.engine.dispose()


__all__ = ["Repository"]

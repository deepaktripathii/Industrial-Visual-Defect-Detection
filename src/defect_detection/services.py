"""Shared service container for API, scripts and dashboard."""

from __future__ import annotations

from dataclasses import dataclass

from defect_detection.config import Settings, get_settings
from defect_detection.db.repository import Repository
from defect_detection.inference.predictor import DefectPredictor, ModelNotReadyError


@dataclass
class Services:
    settings: Settings
    repository: Repository
    predictor: DefectPredictor | None
    model_error: str | None = None

    def close(self) -> None:
        if self.predictor is not None:
            self.predictor.close()
        self.repository.close()


def build_services(
    settings: Settings | None = None,
    *,
    device: str | None = None,
    require_model: bool = False,
) -> Services:
    cfg = settings or get_settings()
    cfg.ensure_directories()

    repository = Repository(settings=cfg)
    try:
        predictor = DefectPredictor(settings=cfg, device=device)
        model_error = None
    except ModelNotReadyError as exc:
        if require_model:
            repository.close()
            raise
        predictor = None
        model_error = str(exc)

    return Services(
        settings=cfg,
        repository=repository,
        predictor=predictor,
        model_error=model_error,
    )


__all__ = ["Services", "build_services"]

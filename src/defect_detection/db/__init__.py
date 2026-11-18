"""Persistence package."""

from defect_detection.db.models import Base, PredictionRecord
from defect_detection.db.repository import Repository

__all__ = ["Base", "PredictionRecord", "Repository"]

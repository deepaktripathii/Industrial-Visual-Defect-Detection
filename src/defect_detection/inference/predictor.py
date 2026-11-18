"""Inference service for single images, batches and Grad-CAM (Features 4 and 6)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from defect_detection.config import NORMAL_CLASS, Settings, get_settings, resolve_device
from defect_detection.explain.gradcam import GradCAM, overlay_png_bytes
from defect_detection.logging_config import get_logger
from defect_detection.modeling.model import load_checkpoint
from defect_detection.preprocessing.transforms import build_eval_transform, open_image

logger = get_logger(__name__)


class ModelNotReadyError(FileNotFoundError):
    """No trained checkpoint is available."""


@dataclass
class Prediction:
    prediction: str
    defect_type: str
    confidence: float
    defective_probability: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model_name: str = "resnet18"
    model_version: str = ""
    image_size: int = 224
    device: str = "cpu"
    latency_ms: float = 0.0
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DefectPredictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        settings: Settings | None = None,
        device: str | None = None,
    ) -> None:
        cfg = settings or get_settings()
        self.settings = cfg
        path = Path(model_path) if model_path else cfg.model_path
        if not path.exists():
            raise ModelNotReadyError(
                f"No model at '{path}'. Train one with `python scripts/train.py` first."
            )

        self.device = resolve_device(device or cfg.device)
        self.model, self.payload = load_checkpoint(path, device=self.device)
        self.model_path = path
        self.classes: list[str] = list(self.payload["classes"])
        self.image_size = int(self.payload.get("image_size", cfg.image_size))
        self.threshold = float(self.payload.get("defective_threshold", cfg.defective_threshold))
        self.version = str(self.payload.get("version", "unknown"))
        self.model_name = str(self.payload.get("model_name", "resnet18"))
        self.transform = build_eval_transform(self.image_size)
        self.normal_index = self.classes.index(NORMAL_CLASS)
        self._gradcam = GradCAM(self.model, self.model.layer4[-1])
        logger.info(
            "Loaded %s (%s) on %s | classes=%s",
            self.model_name,
            self.version,
            self.device,
            self.classes,
        )

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0)

    def _probabilities(self, batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    def _build_prediction(
        self,
        probabilities: np.ndarray,
        *,
        latency_ms: float,
        image: Image.Image | None = None,
    ) -> Prediction:
        index = int(np.argmax(probabilities))
        defective = 1.0 - float(probabilities[self.normal_index])
        is_defective = defective >= self.threshold

        predicted_class = self.classes[index]
        defect_type = predicted_class if predicted_class != NORMAL_CLASS else "none"

        # When the binary decision is "normal", report the normal class as the
        # predicted type even if a defect class has a slightly higher softmax.
        if not is_defective and predicted_class != NORMAL_CLASS:
            defect_type = "none"
            confidence = float(probabilities[self.normal_index])
        else:
            confidence = float(probabilities[index])

        return Prediction(
            prediction="defective" if is_defective else "normal",
            defect_type=defect_type,
            confidence=round(confidence, 4),
            defective_probability=round(defective, 4),
            probabilities={
                name: round(float(probabilities[i]), 4) for i, name in enumerate(self.classes)
            },
            model_name=self.model_name,
            model_version=self.version,
            image_size=self.image_size,
            device=self.device,
            latency_ms=round(latency_ms, 2),
            image_width=image.width if image else 0,
            image_height=image.height if image else 0,
        )

    def predict(self, source: bytes | str | Path | Image.Image) -> Prediction:
        started = time.perf_counter()
        image = open_image(source)
        tensor = self._to_tensor(image)
        probabilities = self._probabilities(tensor)[0]
        return self._build_prediction(
            probabilities, latency_ms=(time.perf_counter() - started) * 1000, image=image
        )

    def predict_bytes(self, data: bytes) -> Prediction:
        return self.predict(data)

    def predict_batch(self, sources: list[bytes | str | Path | Image.Image]) -> list[Prediction]:
        """Single forward pass for several images (throughput optimisation)."""
        if not sources:
            return []

        started = time.perf_counter()
        images = [open_image(source) for source in sources]
        batch = torch.cat([self._to_tensor(image) for image in images], dim=0)
        probabilities = self._probabilities(batch)
        latency_ms = (time.perf_counter() - started) * 1000
        per_image = latency_ms / len(images)
        return [
            self._build_prediction(probabilities[index], latency_ms=per_image, image=images[index])
            for index in range(len(images))
        ]

    # ------------------------------------------------------------------ #
    # Explainability
    # ------------------------------------------------------------------ #
    def explain(
        self,
        source: bytes | str | Path | Image.Image,
        *,
        class_index: int | None = None,
    ) -> tuple[Prediction, bytes]:
        image = open_image(source)
        tensor = self._to_tensor(image)
        prediction = self.predict(image)

        if class_index is None:
            class_index = self.classes.index(
                prediction.defect_type if prediction.defect_type != "none" else NORMAL_CLASS
            )

        cam = self._gradcam.generate(
            tensor, class_index=class_index, out_size=(image.height, image.width)
        )
        png = overlay_png_bytes(image, cam)
        return prediction, png

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "architecture": self.payload.get("arch", "resnet18"),
            "trained_at": self.payload.get("trained_at"),
            "classes": self.classes,
            "image_size": self.image_size,
            "threshold": round(self.threshold, 4),
            "device": self.device,
            "model_path": str(self.model_path),
            "model_size_mb": round(self.model_path.stat().st_size / 1024**2, 2),
            "metrics": self.payload.get("metrics", {}),
            "epoch": self.payload.get("epoch"),
            "normalization": self.payload.get("normalization"),
        }

    def close(self) -> None:
        self._gradcam.remove()


__all__ = ["DefectPredictor", "ModelNotReadyError", "Prediction"]

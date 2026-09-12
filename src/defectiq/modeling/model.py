"""ResNet18 transfer-learning model (Feature 3).

ImageNet weights give the network generic edge/texture filters; the new
classification head is trained on our defect classes. Freezing the backbone for
the first epoch protects those pretrained filters from large early gradients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from defectiq.config import IMAGENET_MEAN, IMAGENET_STD
from defectiq.logging_config import get_logger

logger = get_logger(__name__)

ARCH = "resnet18"


def make_version(prefix: str = ARCH) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d-%H%M%S")
    return f"{prefix}-{stamp}"


def build_model(
    num_classes: int,
    *,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.2,
) -> nn.Module:
    """ResNet18 with a fresh classifier head sized for this dataset."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def parameter_report(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": int(total), "trainable": int(trainable)}


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    classes: list[str],
    image_size: int,
    version: str,
    metrics: dict[str, Any],
    threshold: float,
    arch: str = ARCH,
    extra: dict[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "state_dict": model.state_dict(),
        "classes": list(classes),
        "image_size": int(image_size),
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "model_name": arch,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "defective_threshold": float(threshold),
        **(extra or {}),
    }
    torch.save(payload, destination)
    logger.info("Saved checkpoint %s (%.1f MB)", destination, destination.stat().st_size / 1024**2)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    dropout: float = 0.2,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = list(payload["classes"])
    model = build_model(len(classes), pretrained=False, dropout=dropout)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


__all__ = [
    "ARCH",
    "build_model",
    "load_checkpoint",
    "make_version",
    "parameter_report",
    "save_checkpoint",
]

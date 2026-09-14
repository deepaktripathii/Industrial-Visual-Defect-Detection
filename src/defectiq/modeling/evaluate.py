"""Evaluation: metrics, confusion matrix and model card (Feature 5)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from defectiq.config import NORMAL_CLASS, Settings, get_settings, resolve_device
from defectiq.data.dataset import DatasetInfo, discover_dataset
from defectiq.logging_config import get_logger
from defectiq.modeling.model import load_checkpoint, parameter_report
from defectiq.modeling.trainer import load_history
from defectiq.preprocessing.transforms import build_eval_transform

logger = get_logger(__name__)


def defective_probabilities(
    probabilities: np.ndarray,
    classes: list[str],
    *,
    normal_class: str = NORMAL_CLASS,
) -> np.ndarray:
    """Binary defect score: ``1 * P(good)``, robust for binary and multiclass heads."""
    normal_index = classes.index(normal_class)
    return 1.0 - probabilities[:, normal_index]


def compute_binary_metrics(
    labels_binary: np.ndarray,
    defective_scores: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    predictions = (defective_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_binary, predictions, labels=[0, 1]).ravel()

    auc = None
    if len(np.unique(labels_binary)) > 1:
        auc = round(float(roc_auc_score(labels_binary, defective_scores)), 4)

    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(labels_binary, predictions)), 4),
        "precision": round(float(precision_score(labels_binary, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(labels_binary, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(labels_binary, predictions, zero_division=0)), 4),
        "roc_auc": auc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "support": {
            "normal": int((labels_binary == 0).sum()),
            "defective": int((labels_binary == 1).sum()),
        },
    }


def compute_multiclass_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    indices = list(range(len(classes)))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )
    per_class = [
        {
            "class": name,
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, name in enumerate(classes)
    ]

    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(labels, predictions, average="weighted", zero_division=0)), 4
        ),
        "per_class": per_class,
        "labels": classes,
        "confusion_matrix": confusion_matrix(labels, predictions, labels=indices).tolist(),
    }


@torch.no_grad()
def predict_split(
    model: torch.nn.Module,
    loader,
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return probabilities, labels and mean forward latency per image."""
    model.eval()
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    total = 0
    elapsed = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        started = time.perf_counter()
        logits = model(images)
        elapsed += time.perf_counter() - started
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
        total += labels.size(0)

    if total == 0:
        return np.zeros((0, 0)), np.zeros(0, dtype=int), 0.0
    return np.concatenate(probabilities), np.concatenate(labels_all), (elapsed / total) * 1000


def evaluate_test_split(
    settings: Settings | None = None,
    *,
    model_path: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate the saved checkpoint on an unseen split."""
    from torch.utils.data import DataLoader

    from defectiq.data.dataset import build_image_folder

    cfg = settings or get_settings()
    checkpoint_path = Path(model_path) if model_path else cfg.model_path
    resolved_device = resolve_device(device or cfg.device)

    model, payload = load_checkpoint(checkpoint_path, device=resolved_device)
    classes = list(payload["classes"])
    image_size = int(payload.get("image_size", cfg.image_size))
    threshold = float(payload.get("defective_threshold", cfg.defective_threshold))

    info: DatasetInfo = discover_dataset(cfg.dataset_dir)
    dataset = build_image_folder(
        Path(cfg.dataset_dir) / split, transform=build_eval_transform(image_size)
    )
    if dataset.classes != classes:
        raise ValueError(
            f"Checkpoint classes {classes} do not match {split} split classes {dataset.classes}."
        )

    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    probabilities, labels, latency_ms = predict_split(model, loader, resolved_device)

    predictions = probabilities.argmax(axis=1)
    normal_index = classes.index(NORMAL_CLASS)
    labels_binary = (labels != normal_index).astype(int)
    scores = defective_probabilities(probabilities, classes)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "model_path": str(checkpoint_path),
        "model_name": payload.get("model_name", "resnet18"),
        "model_version": payload.get("version", "unknown"),
        "trained_at": payload.get("trained_at"),
        "classes": classes,
        "image_size": image_size,
        "threshold": threshold,
        "n_images": int(len(labels)),
        "dataset": info.to_dict(),
        "binary": compute_binary_metrics(labels_binary, scores, threshold=threshold),
        "multiclass": compute_multiclass_metrics(labels, predictions, classes),
        "latency": {
            "device": resolved_device,
            "mean_forward_ms_per_image": round(float(latency_ms), 2),
            "batch_size": int(batch_size or cfg.batch_size),
        },
        "training": load_history(cfg.history_path),
        "parameters": parameter_report(model),
        "environment": {
            "torch": torch.__version__,
            "device": resolved_device,
        },
    }
    logger.info(
        "Evaluated %s on %s: binary F1 %.3f, recall %.3f",
        report["model_version"],
        split,
        report["binary"]["f1"],
        report["binary"]["recall"],
    )
    return report


def save_report(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def load_report(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_model_card(report: dict[str, Any], path: str | Path) -> Path:
    card = {
        "model_name": report["model_name"],
        "version": report["model_version"],
        "trained_at": report.get("trained_at"),
        "architecture": "ResNet18 (ImageNet transfer learning)",
        "classes": report["classes"],
        "image_size": report["image_size"],
        "decision_threshold": report["threshold"],
        "parameters": report.get("parameters", {}),
        "dataset": report.get("dataset", {}),
        "performance": {
            "test_binary": report["binary"],
            "test_multiclass": {
                "accuracy": report["multiclass"]["accuracy"],
                "macro_f1": report["multiclass"]["macro_f1"],
                "per_class": report["multiclass"]["per_class"],
            },
            "latency": report["latency"],
        },
        "intended_use": (
            "Flag potentially defective units for human review in a manufacturing "
            "inspection line. The output supports, and never replaces, manual quality "
            "control."
        ),
        "limitations": [
            "Trained on a synthetic surface-defect corpus; fine-tune on the operator's "
            "own camera, lighting and product variants before production use.",
            "Softmax confidence is not calibrated. The 0.5 defect threshold is a "
            "starting point and should be tuned against the cost of misses vs. false alarms.",
            "Grad-CAM highlights influential regions, not verified physical defect masks.",
            "No drift detection on camera conditions yet; monitor rejection rates after "
            "lighting or lens changes.",
        ],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return destination


__all__ = [
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "defective_probabilities",
    "evaluate_test_split",
    "load_report",
    "predict_split",
    "save_report",
    "write_model_card",
]

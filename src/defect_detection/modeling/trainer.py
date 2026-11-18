"""Training loop with transfer learning, class weights and early stopping (Feature 3)."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from defect_detection.config import Settings, get_settings, resolve_device
from defect_detection.data.dataset import (
    DatasetInfo,
    build_image_folder,
    class_weights,
    discover_dataset,
)
from defect_detection.logging_config import get_logger
from defect_detection.modeling.model import (
    build_model,
    make_version,
    parameter_report,
    save_checkpoint,
)
from defect_detection.preprocessing.transforms import build_eval_transform, build_train_transform

logger = get_logger(__name__)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    learning_rate: float
    seconds: float

    def to_dict(self) -> dict[str, float | int]:
        return {key: round(value, 4) if isinstance(value, float) else value for key, value in asdict(self).items()}


@dataclass
class TrainingResult:
    version: str
    model_path: str
    classes: list[str]
    history: list[dict] = field(default_factory=list)
    best_epoch: int = 0
    best_val_f1: float = 0.0
    device: str = "cpu"
    total_seconds: float = 0.0
    image_size: int = 224
    parameters: dict[str, int] = field(default_factory=dict)
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        running_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> dict[str, object]:
    from sklearn.metrics import f1_score

    model.eval()
    running_loss = 0.0
    total = 0
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "probabilities": np.zeros((0, 0))}

    probs = np.concatenate(probabilities, axis=0)
    labels_array = np.concatenate(labels_all, axis=0)
    predictions = probs.argmax(axis=1)
    return {
        "loss": running_loss / total,
        "accuracy": float((predictions == labels_array).mean()),
        "f1_macro": float(f1_score(labels_array, predictions, average="macro", zero_division=0)),
        "probabilities": probs,
        "labels": labels_array,
    }


def save_history(history: list[dict], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return destination


def load_history(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


def train_model(
    settings: Settings | None = None,
    *,
    data_dir: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    image_size: int | None = None,
    pretrained: bool = True,
    freeze_epochs: int | None = None,
    patience: int = 3,
    device: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> TrainingResult:
    """Train ResNet18 on the dataset and save the best checkpoint."""
    cfg = settings or get_settings()
    cfg.ensure_directories()

    root = Path(data_dir) if data_dir else cfg.dataset_dir
    info: DatasetInfo = discover_dataset(root)

    epochs = epochs or cfg.epochs
    batch_size = batch_size or cfg.batch_size
    learning_rate = learning_rate or cfg.learning_rate
    image_size = image_size or cfg.image_size
    freeze_epochs = cfg.freeze_epochs if freeze_epochs is None else freeze_epochs
    resolved_device = resolve_device(device or cfg.device)

    set_seed(cfg.seed)
    version = make_version()

    train_dataset = build_image_folder(
        root / "train", transform=build_train_transform(image_size)
    )
    val_dataset = build_image_folder(root / "val", transform=build_eval_transform(image_size))
    if train_dataset.classes != info.classes:
        raise ValueError(
            f"Train classes {train_dataset.classes} do not match dataset discovery {info.classes}."
        )
    if val_dataset.classes != info.classes:
        raise ValueError("Validation split classes do not match the training split.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=len(train_dataset) % batch_size == 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    model = build_model(
        info.num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_epochs > 0,
    ).to(resolved_device)

    weights = class_weights(info.split_counts["train"], info.classes).to(resolved_device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    if progress:
        progress(
            f"Training {version} on {resolved_device} | {len(train_dataset)} train / "
            f"{len(val_dataset)} val images | {info.num_classes} classes"
        )

    history: list[dict] = []
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.perf_counter()
    stopped_early = False

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()

        if epoch == freeze_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
            if progress:
                progress("Unfroze the backbone for fine-tuning")

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, resolved_device
        )
        validation = evaluate_loader(model, val_loader, criterion, resolved_device)
        scheduler.step()

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=float(validation["loss"]),
            val_accuracy=float(validation["accuracy"]),
            val_f1_macro=float(validation["f1_macro"]),
            learning_rate=float(optimizer.param_groups[0]["lr"]),
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(metrics.to_dict())

        message = (
            f"epoch {epoch:>2}/{epochs} | train loss {metrics.train_loss:.4f} "
            f"acc {metrics.train_accuracy:.3f} | val loss {metrics.val_loss:.4f} "
            f"acc {metrics.val_accuracy:.3f} f1 {metrics.val_f1_macro:.3f} "
            f"| {metrics.seconds:.1f}s"
        )
        logger.info(message)
        if progress:
            progress(message)

        if metrics.val_f1_macro > best_f1 + 1e-4:
            best_f1 = metrics.val_f1_macro
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                cfg.model_path,
                model,
                classes=info.classes,
                image_size=image_size,
                version=version,
                metrics={
                    "val_accuracy": round(metrics.val_accuracy, 4),
                    "val_f1_macro": round(metrics.val_f1_macro, 4),
                    "val_loss": round(metrics.val_loss, 4),
                },
                threshold=cfg.defective_threshold,
                extra={"epoch": epoch, "dataset": info.to_dict()},
            )
        else:
            epochs_without_improvement += 1
            if patience and epochs_without_improvement >= patience:
                stopped_early = True
                if progress:
                    progress(f"Early stopping after {epoch} epochs (patience {patience})")
                break

    if best_epoch == 0:
        # Degenerate validation (e.g. a single class) - still export a usable model.
        save_checkpoint(
            cfg.model_path,
            model,
            classes=info.classes,
            image_size=image_size,
            version=version,
            metrics={},
            threshold=cfg.defective_threshold,
            extra={"dataset": info.to_dict()},
        )
        best_epoch = epochs

    save_history(history, cfg.history_path)

    result = TrainingResult(
        version=version,
        model_path=str(cfg.model_path),
        classes=info.classes,
        history=history,
        best_epoch=best_epoch,
        best_val_f1=round(max(best_f1, 0.0), 4),
        device=resolved_device,
        total_seconds=round(time.perf_counter() - started, 2),
        image_size=image_size,
        parameters=parameter_report(model),
        stopped_early=stopped_early,
    )
    logger.info(
        "Finished %s in %.1fs | best val F1 %.3f (epoch %d)",
        version,
        result.total_seconds,
        result.best_val_f1,
        result.best_epoch,
    )
    return result


__all__ = [
    "EpochMetrics",
    "TrainingResult",
    "evaluate_loader",
    "load_history",
    "save_history",
    "set_seed",
    "train_model",
    "train_one_epoch",
]

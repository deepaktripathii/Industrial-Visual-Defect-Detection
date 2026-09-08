"""Dataset analysis helpers: distributions, image statistics and thumbnails."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from defectiq.data.dataset import IMAGE_SUFFIXES, SPLITS, DatasetInfo
from defectiq.logging_config import get_logger

logger = get_logger(__name__)


def class_distribution(info: DatasetInfo) -> list[dict[str, object]]:
    """Long-form table: split, label, count, share within split."""
    rows: list[dict[str, object]] = []
    for split in SPLITS:
        counts = info.split_counts.get(split, {})
        total = sum(counts.values())
        for label, count in counts.items():
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "count": count,
                    "share": round(count / total, 4) if total else 0.0,
                }
            )
    return rows


def balance_report(info: DatasetInfo) -> dict[str, object]:
    """Imbalance diagnostics for the training split."""
    counts = info.split_counts.get("train", {})
    if not counts:
        return {"balanced": False, "ratio": None, "majority": None, "minority": None}

    majority_label = max(counts, key=counts.get)
    minority_label = min(counts, key=counts.get)
    majority = counts[majority_label]
    minority = counts[minority_label]
    return {
        "balanced": minority > 0 and majority / minority <= 2.0,
        "ratio": round(majority / minority, 2) if minority else None,
        "majority": {"label": majority_label, "count": majority},
        "minority": {"label": minority_label, "count": minority},
    }


def image_size_stats(root: str | Path, *, limit: int = 200) -> dict[str, object]:
    """Sample image dimensions and file sizes."""
    base = Path(root)
    paths = [path for path in base.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES][:limit]
    widths: list[int] = []
    heights: list[int] = []
    sizes: list[int] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                widths.append(image.width)
                heights.append(image.height)
            sizes.append(path.stat().st_size)
        except OSError:  # pragma: no cover - corrupt files are reported elsewhere
            continue

    if not widths:
        return {"sampled": 0}

    return {
        "sampled": len(widths),
        "width": {"min": min(widths), "max": max(widths), "mean": round(float(np.mean(widths)), 1)},
        "height": {"min": min(heights), "max": max(heights), "mean": round(float(np.mean(heights)), 1)},
        "file_kb": {
            "min": round(min(sizes) / 1024, 1),
            "max": round(max(sizes) / 1024, 1),
            "mean": round(float(np.mean(sizes)) / 1024, 1),
        },
    }


def sample_paths(root: str | Path, *, per_class: int = 3, split: str = "train") -> list[Path]:
    base = Path(root) / split
    if not base.is_dir():
        return []
    paths: list[Path] = []
    for class_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        images = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        paths.extend(images[:per_class])
    return paths


def save_preview_grid(
    paths: list[Path],
    destination: str | Path,
    *,
    columns: int = 5,
    thumb_size: int = 160,
) -> Path:
    """Compose a contact sheet of sample images."""
    if not paths:
        raise ValueError("No images to preview.")

    rows = (len(paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size, rows * thumb_size), (16, 20, 30))
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            thumb = image.convert("RGB").resize((thumb_size, thumb_size))
        x = (index % columns) * thumb_size
        y = (index // columns) * thumb_size
        canvas.paste(thumb, (x, y))

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, format="PNG")
    logger.info("Wrote preview grid with %d images to %s", len(paths), target)
    return target


__all__ = [
    "balance_report",
    "class_distribution",
    "image_size_stats",
    "sample_paths",
    "save_preview_grid",
]

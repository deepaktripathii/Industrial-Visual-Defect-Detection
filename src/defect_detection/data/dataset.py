"""Dataset discovery and loading (Feature 1).

Expected layout (ImageFolder style, one folder per class)::

    data/demo_dataset/
      train/good/*.jpg
      train/scratch/*.jpg
      val/good/*.jpg
      test/good/*.jpg
      splits.json          # optional manifest with hashes

The loader is dataset agnostic: any labelled industrial dataset arranged this
way works (synthetic demo set, an operator's own labelled images, or a
classification export derived from MVTec AD test defects).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from defect_detection.config import NORMAL_CLASS
from defect_detection.logging_config import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DatasetLayoutError(ValueError):
    """The dataset folder does not match the expected train/val/test layout."""


@dataclass
class DatasetInfo:
    root: str
    classes: list[str]
    class_to_idx: dict[str, int]
    split_counts: dict[str, dict[str, int]]
    total: int
    normal_class: str = NORMAL_CLASS

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def defective_classes(self) -> list[str]:
        return [name for name in self.classes if name != self.normal_class]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["num_classes"] = self.num_classes
        data["defective_classes"] = self.defective_classes
        return data


def _list_images(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def discover_dataset(root: str | Path) -> DatasetInfo:
    """Validate the layout and summarise classes and counts."""
    base = Path(root)
    if not base.is_dir():
        raise DatasetLayoutError(f"Dataset directory not found: {base}")

    split_classes: dict[str, list[str]] = {}
    for split in SPLITS:
        split_dir = base / split
        if not split_dir.is_dir():
            raise DatasetLayoutError(
                f"Missing '{split}' split in {base}. Expected train/, val/ and test/ folders."
            )
        classes = sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir() and _list_images(path) > 0
        )
        if not classes:
            raise DatasetLayoutError(f"No class folders with images found in {split_dir}.")
        split_classes[split] = classes

    reference = split_classes["train"]
    for split, classes in split_classes.items():
        missing = sorted(set(reference) - set(classes))
        if missing:
            raise DatasetLayoutError(
                f"Split '{split}' is missing classes present in train: {missing}. "
                "Every split must contain the same class folders."
            )
        extra = sorted(set(classes) - set(reference))
        if extra:
            raise DatasetLayoutError(
                f"Split '{split}' contains unexpected classes: {extra}."
            )
    if NORMAL_CLASS not in reference:
        raise DatasetLayoutError(
            f"The defect-free class folder must be called '{NORMAL_CLASS}'."
        )

    split_counts = {
        split: {name: _list_images(base / split / name) for name in reference}
        for split in SPLITS
    }
    total = sum(sum(counts.values()) for counts in split_counts.values())

    info = DatasetInfo(
        root=str(base),
        classes=reference,
        class_to_idx={name: index for index, name in enumerate(reference)},
        split_counts=split_counts,
        total=total,
    )
    logger.info(
        "Dataset %s: %d classes, %d images across %s",
        base,
        info.num_classes,
        info.total,
        ", ".join(f"{split}={sum(counts.values())}" for split, counts in split_counts.items()),
    )
    return info


def build_image_folder(split_root: str | Path, *, transform):
    """Return a torchvision ``ImageFolder`` for one split."""
    folder = datasets.ImageFolder(str(split_root), transform=transform)
    if not folder.classes:
        raise DatasetLayoutError(f"No class folders found in {split_root}")
    return folder


def build_dataloader(
    split_root: str | Path,
    *,
    transform,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = build_image_folder(split_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def class_weights(class_counts: dict[str, int], classes: list[str]):
    """Inverse-frequency class weights for the (possibly imbalanced) training set."""
    import torch

    counts = torch.tensor([max(class_counts.get(name, 0), 1) for name in classes], dtype=torch.float)
    weights = counts.sum() / (len(counts) * counts)
    return weights


__all__ = [
    "DatasetInfo",
    "DatasetLayoutError",
    "IMAGE_SUFFIXES",
    "NORMAL_CLASS",
    "SPLITS",
    "build_dataloader",
    "build_image_folder",
    "class_weights",
    "discover_dataset",
]

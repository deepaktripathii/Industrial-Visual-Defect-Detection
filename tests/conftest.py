"""Shared fixtures: a tiny, fast dataset and a trained checkpoint.

The tiny corpus is drawn with Pillow so tests never need the synthetic generator
script or a real camera dataset. ResNet18 is trained for one epoch without
pretrained weights, which keeps the whole suite offline and quick.
"""

from __future__ import annotations

import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from defectiq.config import get_settings  # noqa: E402
from defectiq.data.splits import save_manifest  # noqa: E402
from defectiq.modeling.evaluate import (  # noqa: E402
    evaluate_test_split,
    save_report,
    write_model_card,
)
from defectiq.modeling.trainer import train_model  # noqa: E402

CLASSES = ("good", "scratch", "dent", "crack", "stain")
SPLIT_SIZES = {"train": 4, "val": 2, "test": 2}
IMAGE_SIZE = 64


def make_tiny_image(label: str, seed: int, size: int = IMAGE_SIZE) -> Image.Image:
    rng = random.Random(seed)
    level = 150 + rng.randint(-12, 12)
    image = Image.new("RGB", (size, size), (level, level, level))
    draw = ImageDraw.Draw(image)

    if label == "scratch":
        y = rng.randint(12, size - 12)
        draw.line([(8, y), (size - 8, y + rng.randint(-6, 6))], fill=(45, 45, 45), width=2)
    elif label == "dent":
        cx, cy = rng.randint(20, size - 20), rng.randint(20, size - 20)
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(60, 62, 65))
    elif label == "crack":
        x, y = 10, rng.randint(16, size - 16)
        points = [(x, y)]
        for _ in range(4):
            x += 12
            y += rng.randint(-8, 8)
            points.append((min(x, size - 6), max(6, min(y, size - 6))))
        draw.line(points, fill=(30, 30, 30), width=2)
    elif label == "stain":
        cx, cy = rng.randint(22, size - 22), rng.randint(22, size - 22)
        draw.ellipse([cx - 12, cy - 10, cx + 12, cy + 10], fill=(90, 78, 68))

    # A seed-derived speck guarantees every image has unique bytes, so the
    # leakage check in the tests only fires on genuine duplicates.
    marker = (level + 25) % 255
    draw.point((seed % size, (seed // size) % size), fill=(marker, marker, marker))
    return image


def build_tiny_dataset(root: Path, *, size: int = IMAGE_SIZE) -> Path:
    for split, per_class in SPLIT_SIZES.items():
        for class_index, label in enumerate(CLASSES):
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                seed = 1000 + class_index * 100 + len(split) * 10 + index
                make_tiny_image(label, seed, size).save(folder / f"{label}_{index}.jpg", quality=92)
    save_manifest(root)
    return root


@pytest.fixture()
def tmp_settings(tmp_path):
    base = get_settings().with_storage(tmp_path)
    settings = replace(
        base,
        image_size=IMAGE_SIZE,
        batch_size=2,
        epochs=1,
        freeze_epochs=0,
        num_workers=0,
        device="cpu",
        defective_threshold=0.5,
    )
    settings.ensure_directories()
    return settings


@pytest.fixture()
def tiny_dataset(tmp_settings):
    return build_tiny_dataset(tmp_settings.dataset_dir)


@pytest.fixture(scope="session")
def trained_settings(tmp_path_factory: pytest.TempPathFactory):
    base = get_settings().with_storage(tmp_path_factory.mktemp("defectiq-artifacts"))
    settings = replace(
        base,
        image_size=IMAGE_SIZE,
        batch_size=2,
        epochs=1,
        freeze_epochs=0,
        num_workers=0,
        device="cpu",
        defective_threshold=0.5,
    )
    settings.ensure_directories()
    build_tiny_dataset(settings.dataset_dir)

    train_model(
        settings,
        pretrained=False,
        epochs=1,
        batch_size=2,
        freeze_epochs=0,
        patience=0,
    )
    report = evaluate_test_split(settings)
    save_report(report, settings.evaluation_path)
    write_model_card(report, settings.model_card_path)
    return settings

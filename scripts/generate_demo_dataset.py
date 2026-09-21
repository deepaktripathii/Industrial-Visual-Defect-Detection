"""Generate a labelled, synthetic surface-defect dataset (Feature 1).

Real industrial datasets (MVTec AD, NEU-DET, DAGM) require registration and
gigabytes of downloads. This script produces a reproducible, clearly labelled
stand-in with five classes - ``good``, ``scratch``, ``dent``, ``crack``,
``stain`` - so the whole pipeline can be trained, evaluated and demonstrated
offline. Swap in a real ImageFolder dataset without changing any code.

Usage::

    python scripts/generate_demo_dataset.py
    python scripts/generate_demo_dataset.py --force --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defectiq.config import get_settings
from defectiq.data.splits import save_manifest

CLASSES = ("good", "scratch", "dent", "crack", "stain")
SPLIT_SIZES = {"train": 40, "val": 10, "test": 14}
IMAGE_SIZE = 224


# --------------------------------------------------------------------------- #
# Surface and defect renderers
# --------------------------------------------------------------------------- #
def make_surface(rng: np.random.Generator, size: int) -> np.ndarray:
    """Brushed metal plate: low-frequency shading + grain + horizontal streaks."""
    base_level = rng.uniform(120, 185)
    surface = rng.normal(base_level, 6, (size, size)).astype(np.float32)

    low = rng.normal(0, 10, (6, 6)).astype(np.float32)
    surface += cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)

    for _ in range(rng.integers(25, 45)):
        y = int(rng.integers(0, size))
        surface[y : y + 1, :] += rng.uniform(-5, 5)

    surface += rng.normal(0, 3.5, (size, size))
    image = np.clip(surface, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def add_scratch(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        x1, y1 = int(rng.integers(10, width - 10)), int(rng.integers(10, height - 10))
        angle = float(rng.uniform(0, np.pi))
        length = int(rng.integers(width // 5, width // 2))
        x2 = int(np.clip(x1 + length * np.cos(angle), 0, width - 1))
        y2 = int(np.clip(y1 + length * np.sin(angle), 0, height - 1))
        shade = int(rng.choice([35, 60, 215, 235]))
        cv2.line(
            image,
            (x1, y1),
            (x2, y2),
            (shade, shade, shade),
            thickness=int(rng.integers(1, 3)),
            lineType=cv2.LINE_AA,
        )
    return image


def add_dent(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    cx = int(rng.integers(width // 4, 3 * width // 4))
    cy = int(rng.integers(height // 4, 3 * height // 4))
    axes = (int(rng.integers(12, 30)), int(rng.integers(12, 30)))
    angle = int(rng.integers(0, 180))

    mask = np.zeros((height, width), np.uint8)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.65

    shadow = np.full_like(image, 55)
    image = (image.astype(np.float32) * (1 - alpha) + shadow.astype(np.float32) * alpha).astype(
        np.uint8
    )
    cv2.ellipse(
        image,
        (cx, cy),
        axes,
        angle,
        200,
        345,
        (205, 210, 215),
        1,
        lineType=cv2.LINE_AA,
    )
    return image


def add_crack(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    start = (int(rng.integers(10, width - 10)), int(rng.integers(10, height - 10)))
    points = [start]
    x, y = start
    for _ in range(int(rng.integers(6, 12))):
        x = int(np.clip(x + rng.integers(-22, 23), 0, width - 1))
        y = int(np.clip(y + rng.integers(-22, 23), 0, height - 1))
        points.append((x, y))

    polyline = np.array(points, dtype=np.int32)
    cv2.polylines(
        image,
        [polyline],
        isClosed=False,
        color=(30, 32, 35),
        thickness=int(rng.integers(2, 4)),
        lineType=cv2.LINE_AA,
    )
    # Branches make cracks look less like straight scratches.
    for _ in range(int(rng.integers(1, 3))):
        if len(points) <= 3:
            break
        anchor = points[int(rng.integers(1, len(points) - 1))]
        branch_end = (
            int(np.clip(anchor[0] + rng.integers(-38, 39), 0, width - 1)),
            int(np.clip(anchor[1] + rng.integers(-38, 39), 0, height - 1)),
        )
        cv2.line(image, anchor, branch_end, (35, 37, 40), 2, lineType=cv2.LINE_AA)
    return image


def add_stain(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    cx = int(rng.integers(width // 4, 3 * width // 4))
    cy = int(rng.integers(height // 4, 3 * height // 4))
    radius = int(rng.integers(18, 42))

    points = []
    for _ in range(10):
        angle = float(rng.uniform(0, 2 * np.pi))
        distance = radius * float(rng.uniform(0.55, 1.0))
        points.append(
            (
                int(np.clip(cx + distance * np.cos(angle), 0, width - 1)),
                int(np.clip(cy + distance * np.sin(angle), 0, height - 1)),
            )
        )

    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * 0.65

    stain_color = np.array([50, 68, 88], dtype=np.float32)  # BGR, oily brown
    image = (image.astype(np.float32) * (1 - alpha) + stain_color * alpha).astype(np.uint8)
    return image


DEFECT_RENDERERS = {
    "scratch": add_scratch,
    "dent": add_dent,
    "crack": add_crack,
    "stain": add_stain,
}


def generate_image(label: str, rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    image = make_surface(rng, size)
    renderer = DEFECT_RENDERERS.get(label)
    if renderer is not None:
        image = renderer(image, rng)
    return image


# --------------------------------------------------------------------------- #
# Dataset creation
# --------------------------------------------------------------------------- #
def generate_dataset(
    root: Path,
    *,
    seed: int = 42,
    force: bool = False,
    size: int = IMAGE_SIZE,
    progress=None,
) -> dict[str, int]:
    if root.exists() and any(root.glob("*/**/*.jpg")) and not force:
        raise FileExistsError(f"{root} already contains images. Re-run with --force to replace.")

    total = 0
    for split, per_class in SPLIT_SIZES.items():
        for class_index, label in enumerate(CLASSES):
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(per_class):
                image_seed = seed + class_index * 100_000 + len(split) * 10_000 + index * 17
                rng = np.random.default_rng(image_seed)
                image = generate_image(label, rng, size)
                path = folder / f"{label}_{index:03d}.jpg"
                cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
                total += 1
        if progress:
            progress(f"{split}: {per_class * len(CLASSES)} images")

    manifest_path = save_manifest(root)
    return {"total": total, "manifest": str(manifest_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic defect dataset.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    root = Path(args.output_dir) if args.output_dir else settings.dataset_dir
    print(f"Generating synthetic surface defects in {root}")
    result = generate_dataset(
        root,
        seed=args.seed,
        force=args.force,
        size=args.size,
        progress=lambda message: print(f"  {message}"),
    )
    print(f"Done: {result['total']} images · manifest {result['manifest']}")
    print("Next: python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

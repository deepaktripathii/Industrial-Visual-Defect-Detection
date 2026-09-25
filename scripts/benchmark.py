"""Latency benchmark: batch size and device comparison (Feature 8).

Usage::

    python scripts/benchmark.py
    python scripts/benchmark.py --batches 1 4 8 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defectiq.config import get_settings
from defectiq.inference.predictor import DefectPredictor
from defectiq.preprocessing.transforms import open_image


def _sample_image(settings) -> Path:
    candidates = sorted((settings.dataset_dir / "test").rglob("*.jpg"))
    if not candidates:
        raise SystemExit(
            "No test images found. Run `python scripts/generate_demo_dataset.py` first."
        )
    return candidates[0]


def measure(predictor: DefectPredictor, image_path: Path, batch_size: int, runs: int = 5) -> dict:
    tensor = predictor.transform(open_image(image_path)).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    tensor = tensor.to(predictor.device)

    with torch.no_grad():
        predictor.model(tensor)  # warm-up
        timings = []
        for _ in range(runs):
            started = time.perf_counter()
            predictor.model(tensor)
            timings.append((time.perf_counter() - started) * 1000)

    mean_ms = sum(timings) / len(timings)
    return {
        "batch_size": batch_size,
        "device": predictor.device,
        "mean_batch_ms": round(mean_ms, 2),
        "ms_per_image": round(mean_ms / batch_size, 2),
        "throughput_images_per_second": round(batch_size / (mean_ms / 1000), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark model inference latency.")
    parser.add_argument("--image", default=None)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args(argv)

    settings = get_settings()
    image_path = Path(args.image) if args.image else _sample_image(settings)

    predictor = DefectPredictor(settings=settings)
    print(f"Model   : {predictor.version}")
    print(f"Device  : {predictor.device}")
    print(f"Image   : {image_path}")

    rows = []
    try:
        for batch_size in args.batches:
            rows.append(measure(predictor, image_path, batch_size, runs=args.runs))

        end_to_end = []
        for _ in range(args.runs):
            started = time.perf_counter()
            predictor.predict(image_path)
            end_to_end.append((time.perf_counter() - started) * 1000)
        single_image_ms = sum(end_to_end) / len(end_to_end)
    finally:
        predictor.close()

    print("\nBatch inference")
    print(f"{'batch':>6} {'batch ms':>10} {'ms/image':>10} {'images/s':>10}")
    for row in rows:
        print(
            f"{row['batch_size']:>6} {row['mean_batch_ms']:>10.1f} "
            f"{row['ms_per_image']:>10.1f} {row['throughput_images_per_second']:>10.1f}"
        )

    print(f"\nEnd-to-end single image (decode + preprocess + forward): {single_image_ms:.1f} ms")

    report = {
        "model_version": predictor.version if predictor else None,
        "device": rows[0]["device"] if rows else None,
        "single_image_ms": round(single_image_ms, 2),
        "batches": rows,
    }
    destination = settings.report_dir / "benchmark.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

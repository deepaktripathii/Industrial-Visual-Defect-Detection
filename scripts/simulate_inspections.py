"""Generate demo inspection traffic for the monitoring dashboard.

Usage::

    python scripts/simulate_inspections.py --n 60
"""

from __future__ import annotations

import argparse
import hashlib
import random

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defectiq.config import get_settings
from defectiq.db.repository import Repository
from defectiq.inference.predictor import DefectPredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log demo inspections from the test split.")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--defective-only", action="store_true")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args(argv)

    settings = get_settings()
    test_root = settings.dataset_dir / "test"
    if not test_root.is_dir():
        print("No test split found. Run `python scripts/generate_demo_dataset.py` first.")
        return 1

    images = sorted(test_root.rglob("*.jpg"))
    if args.defective_only:
        images = [path for path in images if path.parent.name != "good"]
    if not images:
        print("No matching images.")
        return 1

    random.seed(args.seed)
    sample = random.sample(images, k=min(args.n, len(images)))

    predictor = DefectPredictor(settings=settings, device=args.device)
    repository = Repository(settings=settings)
    counts: dict[str, int] = {}
    try:
        for path in sample:
            data = path.read_bytes()
            prediction = predictor.predict_bytes(data)
            repository.log_prediction(
                filename=path.name,
                image_sha256=hashlib.sha256(data).hexdigest(),
                content_type="image/jpeg",
                size_bytes=len(data),
                image_width=prediction.image_width,
                image_height=prediction.image_height,
                prediction=prediction.prediction,
                defect_type=prediction.defect_type,
                confidence=prediction.confidence,
                defective_probability=prediction.defective_probability,
                model_name=prediction.model_name,
                model_version=prediction.model_version,
                latency_ms=prediction.latency_ms,
                device=prediction.device,
                status="success",
            )
            counts[prediction.prediction] = counts.get(prediction.prediction, 0) + 1
    finally:
        predictor.close()
        repository.close()

    print(f"Logged {len(sample)} inspections with {predictor.version}")
    print(f"Predictions: {counts}")
    print(f"Database   : {settings.database_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

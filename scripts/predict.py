"""Run inference on a single image from the command line.

Usage::

    python scripts/predict.py data/demo_dataset/test/scratch/scratch_000.jpg
    python scripts/predict.py image.png --explain artifacts/reports/explain.png
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defect_detection.config import get_settings
from defect_detection.inference.predictor import DefectPredictor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect one image with the saved model.")
    parser.add_argument("image", help="Path to a JPEG/PNG image.")
    parser.add_argument("--model", default=None, help="Checkpoint override.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--explain",
        nargs="?",
        const="artifacts/reports/explain.png",
        default=None,
        help="Write a Grad-CAM overlay to this path (default artifacts/reports/explain.png).",
    )
    parser.add_argument("--json-base64", action="store_true", help="Include the overlay as base64.")
    args = parser.parse_args(argv)

    settings = get_settings()
    predictor = DefectPredictor(args.model, settings=settings, device=args.device)

    try:
        if args.explain:
            prediction, png = predictor.explain(args.image)
            destination = Path(args.explain)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(png)
        else:
            prediction = predictor.predict(args.image)
            png = None
    finally:
        predictor.close()

    payload = prediction.to_dict()
    if png and args.json_base64:
        payload["gradcam_png_base64"] = base64.b64encode(png).decode("ascii")
    print(json.dumps(payload, indent=2))

    if args.explain:
        print(f"\nGrad-CAM overlay written to {Path(args.explain)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

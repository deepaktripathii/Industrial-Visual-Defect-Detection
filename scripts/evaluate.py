"""Evaluate a saved checkpoint on the val or test split.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --split val --json
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defect_detection.config import get_settings
from defect_detection.modeling.evaluate import (
    evaluate_test_split,
    save_report,
    write_model_card,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the defect classifier.")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model", default=None, help="Checkpoint path override.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--json", action="store_true", help="Print the full report.")
    args = parser.parse_args(argv)

    settings = get_settings()
    report = evaluate_test_split(
        settings,
        model_path=args.model,
        batch_size=args.batch_size,
        device=args.device,
        split=args.split,
    )
    save_report(report, settings.evaluation_path)
    write_model_card(report, settings.model_card_path)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    binary = report["binary"]
    print(f"Model    : {report['model_version']} ({report['model_name']})")
    print(f"Split    : {report['split']} ({report['n_images']} images)")
    print(f"Device   : {report['latency']['device']}")
    print("\nBinary defect detection")
    print(f"  accuracy        : {binary['accuracy']:.3f}")
    print(f"  precision       : {binary['precision']:.3f}")
    print(f"  recall          : {binary['recall']:.3f}")
    print(f"  F1              : {binary['f1']:.3f}")
    print(f"  ROC AUC         : {binary['roc_auc']}")
    print(f"  threshold       : {binary['threshold']:.2f}")
    matrix = binary["confusion_matrix"]
    print(
        "  confusion       : "
        f"TP={matrix['true_positive']} FP={matrix['false_positive']} "
        f"FN={matrix['false_negative']} TN={matrix['true_negative']}"
    )

    print("\nPer-class performance")
    per_class = pd.DataFrame(report["multiclass"]["per_class"])
    print(per_class.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print(f"\nMean forward latency: {report['latency']['mean_forward_ms_per_image']:.1f} ms/image")
    print(f"Report written to {settings.evaluation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

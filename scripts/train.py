"""Train the ResNet18 defect classifier and evaluate it on the test split.

Usage::

    python scripts/train.py
    python scripts/train.py --epochs 12 --batch-size 16
    python scripts/train.py --no-pretrained --epochs 3       # offline / CI
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401  (adds src/ to sys.path)
from defect_detection.config import get_settings
from defect_detection.data.dataset import discover_dataset
from defect_detection.data.splits import load_manifest, verify_manifest
from defect_detection.modeling.evaluate import evaluate_test_split, save_report, write_model_card
from defect_detection.modeling.trainer import train_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Visual Defect Detection classifier.")
    parser.add_argument("--data-dir", default=None, help="Dataset root (train/val/test).")
    parser.add_argument("--artifacts", default=None, help="Artifact directory.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Skip ImageNet weights (faster, offline, lower accuracy).",
    )
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    if args.artifacts:
        settings = settings.with_artifacts(args.artifacts)
    if args.data_dir:
        settings = replace(settings, dataset_dir=Path(args.data_dir).expanduser().resolve())
    settings.ensure_directories()

    info = discover_dataset(settings.dataset_dir)
    print(f"Dataset : {settings.dataset_dir}")
    print(
        "Classes : "
        + ", ".join(f"{name}={sum(split.get(name, 0) for split in info.split_counts.values())}" for name in info.classes)
    )
    table = pd.DataFrame(
        [
            {"split": split, "label": label, "count": count}
            for split, counts in info.split_counts.items()
            for label, count in counts.items()
        ]
    )
    print(table.pivot(index="label", columns="split", values="count").to_string())

    if settings.splits_manifest_path.exists():
        report = verify_manifest(load_manifest(settings.splits_manifest_path))
        if report["duplicates"]:
            print("\n[warning] Duplicate images across splits - this is data leakage:")
            for duplicate in report["duplicates"]:
                print(f"  {duplicate['splits']}: {duplicate['paths']}")
        else:
            print(f"\nSplit manifest OK: {report['n_files']} unique images, no cross-split duplicates.")

    result = train_model(
        settings,
        data_dir=settings.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        freeze_epochs=args.freeze_epochs,
        patience=args.patience,
        device=args.device,
        pretrained=not args.no_pretrained,
        progress=print,
    )

    print(
        f"\nBest validation macro-F1: {result.best_val_f1:.3f} "
        f"(epoch {result.best_epoch}, {result.total_seconds:.1f}s on {result.device})"
    )
    print(f"Checkpoint: {result.model_path}")

    if not args.skip_eval:
        report = evaluate_test_split(settings, device=args.device)
        save_report(report, settings.evaluation_path)
        write_model_card(report, settings.model_card_path)
        binary = report["binary"]
        print("\nTest set")
        print(f"  binary accuracy : {binary['accuracy']:.3f}")
        print(f"  precision       : {binary['precision']:.3f}")
        print(f"  recall          : {binary['recall']:.3f}")
        print(f"  F1              : {binary['f1']:.3f}")
        print(f"  ROC AUC         : {binary['roc_auc']}")
        print(f"  macro F1 (5-way): {report['multiclass']['macro_f1']:.3f}")
        print(f"  latency         : {report['latency']['mean_forward_ms_per_image']:.1f} ms/image")
        print(f"\nReport : {settings.evaluation_path}")
        print(f"Card   : {settings.model_card_path}")

    print("\nNext: uvicorn defect_detection.api.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

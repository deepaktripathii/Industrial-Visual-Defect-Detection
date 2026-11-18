"""Split manifests and leakage checks (Feature 1).

The generator writes ``splits.json`` with a SHA-256 hash per image. Verification
answers the question every reviewer asks: *is the same image present in both the
training and test split?* That is the most common dataset leakage bug in vision
projects.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from defect_detection.data.dataset import IMAGE_SUFFIXES, SPLITS
from defect_detection.logging_config import get_logger

logger = get_logger(__name__)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def compute_manifest(root: str | Path) -> dict[str, Any]:
    """Hash every image and record its split and label."""
    base = Path(root)
    files: list[dict[str, Any]] = []
    for split in SPLITS:
        split_dir = base / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                files.append(
                    {
                        "path": image_path.relative_to(base).as_posix(),
                        "split": split,
                        "label": class_dir.name,
                        "sha256": sha256_file(image_path),
                        "size_bytes": image_path.stat().st_size,
                    }
                )

    return {
        "root": str(base),
        "n_files": len(files),
        "files": files,
    }


def save_manifest(root: str | Path, destination: str | Path | None = None) -> Path:
    base = Path(root)
    target = Path(destination) if destination else base / "splits.json"
    manifest = compute_manifest(base)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote split manifest with %d images to %s", manifest["n_files"], target)
    return target


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check for duplicates across splits and for missing files.

    Returns a report with ``ok``, ``duplicates``, ``missing`` and per-split counts.
    """
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[str] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for entry in manifest.get("files", []):
        by_hash[entry["sha256"]].append(entry)
        counts[entry["split"]][entry["label"]] += 1

    root = Path(manifest.get("root", ""))
    for entry in manifest.get("files", []):
        if root and not (root / entry["path"]).exists():
            missing.append(entry["path"])

    duplicates = []
    for digest, entries in by_hash.items():
        splits = {entry["split"] for entry in entries}
        if len(entries) > 1 and len(splits) > 1:
            duplicates.append(
                {
                    "sha256": digest,
                    "paths": [entry["path"] for entry in entries],
                    "splits": sorted(splits),
                }
            )

    report = {
        "ok": not duplicates and not missing,
        "n_files": len(manifest.get("files", [])),
        "duplicates": duplicates,
        "missing": missing,
        "counts": {split: dict(values) for split, values in counts.items()},
    }
    return report


__all__ = [
    "compute_manifest",
    "load_manifest",
    "save_manifest",
    "sha256_file",
    "verify_manifest",
]

"""Feature 1: dataset discovery, class weights and split leakage checks."""

from __future__ import annotations

import shutil

import pytest

from defectiq.data.analysis import balance_report, class_distribution
from defectiq.data.dataset import DatasetLayoutError, class_weights, discover_dataset
from defectiq.data.splits import compute_manifest, load_manifest, verify_manifest


def test_discover_dataset_counts(tiny_dataset):
    info = discover_dataset(tiny_dataset)

    assert info.classes == ["crack", "dent", "good", "scratch", "stain"]
    assert info.num_classes == 5
    assert info.total == 40
    assert info.split_counts["train"]["good"] == 4
    assert info.defective_classes == ["crack", "dent", "scratch", "stain"]


def test_missing_split_raises(tiny_dataset):
    shutil.rmtree(tiny_dataset / "val")

    with pytest.raises(DatasetLayoutError, match="val"):
        discover_dataset(tiny_dataset)


def test_missing_normal_class_raises(tmp_path):
    for split in ("train", "val", "test"):
        (tmp_path / split / "scratch").mkdir(parents=True)
        (tmp_path / split / "scratch" / "a.jpg").write_bytes(b"x")

    with pytest.raises(DatasetLayoutError, match="good"):
        discover_dataset(tmp_path)


def test_class_weights_balance_imbalance():
    weights = class_weights({"good": 10, "scratch": 2}, ["good", "scratch"])

    assert weights.shape == (2,)
    assert weights[1] / weights[0] == pytest.approx(5.0)
    assert float(weights.sum()) == pytest.approx(3.6)


def test_class_distribution_and_balance(tiny_dataset):
    info = discover_dataset(tiny_dataset)
    rows = class_distribution(info)
    balance = balance_report(info)

    assert len(rows) == 15
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert balance["balanced"] is True


def test_manifest_has_no_leakage(tiny_dataset):
    report = verify_manifest(load_manifest(tiny_dataset / "splits.json"))

    assert report["ok"] is True
    assert report["n_files"] == 40
    assert report["duplicates"] == []


def test_manifest_detects_cross_split_duplicate(tiny_dataset):
    source = tiny_dataset / "train" / "good" / "good_0.jpg"
    leaked = tiny_dataset / "test" / "good" / "leaked.jpg"
    shutil.copyfile(source, leaked)
    manifest = compute_manifest(tiny_dataset)

    report = verify_manifest(manifest)

    assert report["ok"] is False
    assert len(report["duplicates"]) == 1
    assert report["duplicates"][0]["splits"] == ["test", "train"]


def test_manifest_detects_missing_file(tiny_dataset):
    manifest = compute_manifest(tiny_dataset)
    (tiny_dataset / "train" / "good" / "good_0.jpg").unlink()

    report = verify_manifest(manifest)

    assert report["ok"] is False
    assert report["missing"]

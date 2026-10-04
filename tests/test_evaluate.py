"""Feature 5: metrics, confusion matrix and the test-set report."""

from __future__ import annotations

import numpy as np
import pytest

from defectiq.modeling.evaluate import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    evaluate_test_split,
)


def test_binary_metrics_perfect_prediction():
    metrics = compute_binary_metrics(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == {
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_positive": 2,
    }


def test_threshold_trades_recall_for_precision():
    labels = np.array([0, 1])
    scores = np.array([0.6, 0.6])

    at_half = compute_binary_metrics(labels, scores, threshold=0.5)
    at_seventy = compute_binary_metrics(labels, scores, threshold=0.7)

    assert at_half["confusion_matrix"]["false_positive"] == 1
    assert at_half["confusion_matrix"]["true_positive"] == 1
    assert at_seventy["confusion_matrix"]["false_positive"] == 0
    assert at_seventy["confusion_matrix"]["false_negative"] == 1
    assert at_half["recall"] > at_seventy["recall"]


def test_multiclass_metrics_per_class():
    labels = np.array([0, 0, 1, 1, 2])
    predictions = np.array([0, 0, 1, 2, 2])

    metrics = compute_multiclass_metrics(labels, predictions, ["good", "scratch", "stain"])

    assert metrics["accuracy"] == pytest.approx(0.8)
    assert len(metrics["per_class"]) == 3
    assert metrics["per_class"][0]["recall"] == 1.0
    assert metrics["per_class"][1]["recall"] == pytest.approx(0.5)
    assert len(metrics["confusion_matrix"]) == 3


def test_evaluate_test_split(trained_settings):
    report = evaluate_test_split(trained_settings)

    assert report["n_images"] == 10
    assert set(report["binary"]) >= {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert len(report["multiclass"]["per_class"]) == 5
    assert len(report["multiclass"]["confusion_matrix"]) == 5
    assert report["latency"]["device"] == "cpu"
    assert report["latency"]["mean_forward_ms_per_image"] >= 0
    assert report["model_version"].startswith("resnet18-")
    assert report["training"]

"""Feature 8: inspection history and monitoring aggregates."""

from __future__ import annotations

import pytest

from defect_detection.db.repository import Repository


@pytest.fixture()
def repository(tmp_path):
    instance = Repository(f"sqlite:///{(tmp_path / 'monitoring.db').as_posix()}")
    try:
        yield instance
    finally:
        instance.close()


def _log(repository: Repository, *, prediction: str = "normal", defect: str = "none", latency: float = 20.0):
    return repository.log_prediction(
        filename=f"{prediction}.jpg",
        image_sha256="a" * 64,
        size_bytes=1024,
        image_width=224,
        image_height=224,
        prediction=prediction,
        defect_type=defect,
        confidence=0.9,
        defective_probability=0.1 if prediction == "normal" else 0.9,
        model_name="resnet18",
        model_version="resnet18-test",
        latency_ms=latency,
        device="cpu",
    )


def test_log_and_recent(repository):
    record = _log(repository, prediction="defective", defect="scratch")

    assert record["id"] == 1
    rows = repository.recent(limit=5)
    assert len(rows) == 1
    assert rows[0]["defect_type"] == "scratch"
    assert repository.count() == 1


def test_filter_by_prediction(repository):
    _log(repository, prediction="normal")
    _log(repository, prediction="defective", defect="crack")

    defective = repository.recent(limit=10, prediction="defective")

    assert len(defective) == 1
    assert defective[0]["prediction"] == "defective"


def test_summary_counts_and_latency(repository):
    _log(repository, prediction="normal", latency=10)
    _log(repository, prediction="defective", defect="dent", latency=30)
    repository.log_prediction(
        filename="broken.jpg",
        prediction="normal",
        defect_type="none",
        confidence=0.0,
        defective_probability=0.0,
        model_name="resnet18",
        model_version="resnet18-test",
        latency_ms=5,
        status="failed",
        error="bad image",
    )

    summary = repository.summary()

    assert summary["total_inspections"] == 3
    assert summary["normal_count"] == 1
    assert summary["defective_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["defect_types"] == {"dent": 1}
    assert summary["avg_latency_ms"] == pytest.approx(15.0)
    assert summary["defect_rate"] == pytest.approx(0.5)
    assert summary["model_versions"] == {"resnet18-test": 3}


def test_empty_summary(repository):
    summary = repository.summary()

    assert summary["total_inspections"] == 0
    assert summary["defect_rate"] is None
    assert summary["daily"] == []


def test_clear(repository):
    _log(repository)
    repository.clear()

    assert repository.count() == 0

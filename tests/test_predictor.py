"""Features 4 and 6: single, batch and explained inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from defect_detection.inference.predictor import DefectPredictor, ModelNotReadyError


@pytest.fixture()
def predictor(trained_settings):
    instance = DefectPredictor(settings=trained_settings)
    try:
        yield instance
    finally:
        instance.close()


def _image(settings, label: str = "good") -> Path:
    return next((Path(settings.dataset_dir) / "test" / label).glob("*.jpg"))


def test_predict_returns_probabilities(predictor, trained_settings):
    prediction = predictor.predict(_image(trained_settings))

    assert prediction.prediction in {"normal", "defective"}
    assert 0.0 <= prediction.confidence <= 1.0
    assert 0.0 <= prediction.defective_probability <= 1.0
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert prediction.model_version.startswith("resnet18-")
    assert prediction.image_width == 64
    assert prediction.latency_ms >= 0


def test_binary_decision_follows_threshold(predictor, trained_settings):
    prediction = predictor.predict(_image(trained_settings, "scratch"))

    expected = "defective" if prediction.defective_probability >= predictor.threshold else "normal"
    assert prediction.prediction == expected
    if prediction.prediction == "normal":
        assert prediction.defect_type == "none"


def test_batch_matches_single_predictions(predictor, trained_settings):
    paths = [_image(trained_settings, "good"), _image(trained_settings, "crack")]

    batch = predictor.predict_batch(paths)
    single = [predictor.predict(path) for path in paths]

    assert len(batch) == 2
    assert [item.prediction for item in batch] == [item.prediction for item in single]
    for item in batch:
        assert item.latency_ms >= 0


def test_explain_returns_png(predictor, trained_settings):
    prediction, png = predictor.explain(_image(trained_settings, "stain"))

    assert prediction.prediction in {"normal", "defective"}
    assert png.startswith(b"\x89PNG")
    assert len(png) > 500


def test_info_reports_model_metadata(predictor):
    info = predictor.info()

    assert info["model_name"] == "resnet18"
    assert len(info["classes"]) == 5
    assert info["image_size"] == 64
    assert info["device"] == "cpu"
    assert info["model_size_mb"] > 0
    assert info["threshold"] == 0.5


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ModelNotReadyError, match="Train one"):
        DefectPredictor(tmp_path / "missing.pt")


def test_empty_batch(predictor):
    assert predictor.predict_batch([]) == []

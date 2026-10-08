"""Feature 7: FastAPI endpoints and inspection logging."""

from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from defectiq.api.main import create_app


@pytest.fixture()
def client(trained_settings, tmp_path):
    settings = replace(
        trained_settings,
        database_url=f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _image(settings, label: str = "good") -> Path:
    return next((Path(settings.dataset_dir) / "test" / label).glob("*.jpg"))


def _upload(client: TestClient, path: Path, *, explain: bool = False):
    return client.post(
        "/predict",
        params={"explain": explain},
        files={"file": (path.name, path.read_bytes(), "image/jpeg")},
    )


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "resnet18"
    assert body["device"] == "cpu"


def test_root_and_model_info(client):
    root = client.get("/").json()
    assert root["docs"] == "/docs"

    info = client.get("/model-info")
    assert info.status_code == 200
    body = info.json()
    assert len(body["classes"]) == 5
    assert body["image_size"] == 64
    assert body["architecture"] == "resnet18"


def test_predict_logs_inspection(client, trained_settings):
    response = _upload(client, _image(trained_settings, "scratch"))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction"] in {"normal", "defective"}
    assert 0 <= body["confidence"] <= 1
    assert body["id"] is not None
    assert body["heatmap_png_base64"] is None
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    summary = client.get("/monitoring/summary").json()
    assert summary["total_inspections"] == 1


def test_predict_with_gradcam(client, trained_settings):
    response = _upload(client, _image(trained_settings, "crack"), explain=True)

    assert response.status_code == 200
    encoded = response.json()["heatmap_png_base64"]
    assert encoded
    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_batch_prediction(client, trained_settings):
    paths = [_image(trained_settings, "good"), _image(trained_settings, "stain")]
    files = [("files", (path.name, path.read_bytes(), "image/jpeg")) for path in paths]

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert client.get("/monitoring/summary").json()["total_inspections"] == 2


def test_invalid_image_is_rejected_and_logged(client):
    response = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"not an image at all", "image/jpeg")},
    )

    assert response.status_code == 400
    summary = client.get("/monitoring/summary").json()
    assert summary["failure_count"] == 1
    assert summary["total_inspections"] == 1


def test_predictions_history(client, trained_settings):
    _upload(client, _image(trained_settings, "good"))
    _upload(client, _image(trained_settings, "stain"))

    rows = client.get("/predictions", params={"limit": 10}).json()

    assert len(rows) == 2
    assert rows[0]["filename"]


def test_evaluation_endpoints(client):
    latest = client.get("/evaluation/latest")
    assert latest.status_code == 200
    assert latest.json()["split"] == "test"

    run = client.post("/evaluation/run", params={"split": "test"})
    assert run.status_code == 200
    assert run.json()["n_images"] == 10


def test_unsupported_batch_size(client, trained_settings):
    path = _image(trained_settings, "good")
    files = [("files", (path.name, path.read_bytes(), "image/jpeg"))] * 17

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 400

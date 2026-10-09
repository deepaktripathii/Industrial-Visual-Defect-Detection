"""DefectIQ Streamlit dashboard.

Run with::

    streamlit run src/defectiq/dashboard/app.py

Uses the FastAPI service when it is reachable and falls back to local inference
and the local database, so the dashboard also works standalone.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402
import requests  # noqa: E402
import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from defectiq.config import get_settings  # noqa: E402
from defectiq.data.analysis import (  # noqa: E402
    balance_report,
    class_distribution,
    sample_paths,
    save_preview_grid,
)
from defectiq.data.dataset import discover_dataset  # noqa: E402
from defectiq.data.splits import load_manifest, verify_manifest  # noqa: E402
from defectiq.db.repository import Repository  # noqa: E402
from defectiq.inference.predictor import DefectPredictor, ModelNotReadyError  # noqa: E402
from defectiq.modeling.evaluate import load_report  # noqa: E402

SETTINGS = get_settings()

st.set_page_config(
    page_title="DefectIQ | Visual Inspection",
    page_icon="🔍",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def evaluation_report() -> dict | None:
    return load_report(SETTINGS.evaluation_path)


@st.cache_data(show_spinner=False)
def training_history() -> list[dict]:
    path = SETTINGS.history_path
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def model_card() -> dict | None:
    path = SETTINGS.model_card_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def benchmark_report() -> dict | None:
    path = SETTINGS.report_dir / "benchmark.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def dataset_overview():
    try:
        info = discover_dataset(SETTINGS.dataset_dir)
    except Exception:
        return None, [], {}
    return info.to_dict(), class_distribution(info), balance_report(info)


@st.cache_data(show_spinner=False)
def dataset_preview() -> bytes | None:
    paths = sample_paths(SETTINGS.dataset_dir, per_class=2, split="test")
    if not paths:
        return None
    destination = SETTINGS.report_dir / "dataset_preview.png"
    save_preview_grid(paths, destination, columns=5, thumb_size=150)
    return destination.read_bytes()


@st.cache_data(show_spinner=False)
def split_check() -> dict | None:
    manifest_path = SETTINGS.splits_manifest_path
    if not manifest_path.exists():
        return None
    return verify_manifest(load_manifest(manifest_path))


@st.cache_resource(show_spinner=False)
def local_predictor() -> DefectPredictor | None:
    try:
        return DefectPredictor(settings=SETTINGS)
    except ModelNotReadyError:
        return None


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api_get(api_url: str, path: str, params: dict | None = None):
    try:
        response = requests.get(f"{api_url.rstrip('/')}{path}", params=params, timeout=4)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def api_predict(api_url: str, data: bytes, filename: str, *, explain: bool):
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            params={"explain": explain},
            files={"file": (filename, data, "image/jpeg")},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def local_inspect(data: bytes, filename: str, *, explain: bool) -> dict | None:
    model = local_predictor()
    if model is None:
        return None

    if explain:
        prediction, png = model.explain(data)
        payload = prediction.to_dict()
        payload["heatmap_png_base64"] = base64.b64encode(png).decode("ascii")
    else:
        payload = model.predict_bytes(data).to_dict()

    repository = Repository(settings=SETTINGS)
    try:
        repository.log_prediction(
            filename=filename,
            image_sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
            image_width=payload["image_width"],
            image_height=payload["image_height"],
            prediction=payload["prediction"],
            defect_type=payload["defect_type"],
            confidence=payload["confidence"],
            defective_probability=payload["defective_probability"],
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            latency_ms=payload["latency_ms"],
            device=payload["device"],
        )
    finally:
        repository.close()
    return payload


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔍 DefectIQ")
    st.caption("Visual quality inspection")
    page = st.radio(
        "Navigate",
        ["Inspect", "Model performance", "Monitoring", "Dataset"],
        label_visibility="collapsed",
    )
    st.divider()
    api_url = st.text_input(
        "API base URL",
        value=os.getenv("DEFECTIQ_API_URL", "http://localhost:8000"),
        help="Used for predictions and monitoring; local inference is the fallback.",
    )
    health = api_get(api_url, "/health")
    if health:
        st.success(f"API online · {health.get('model_version') or 'no model'}")
    else:
        st.warning("API offline · local inference")
    st.divider()
    st.caption(f"Model: `{SETTINGS.model_path.name}`")


# --------------------------------------------------------------------------- #
# Inspect
# --------------------------------------------------------------------------- #
def render_inspect() -> None:
    st.header("Inspect a product image")
    st.caption(
        "Upload a surface image. The model returns normal/defective, the predicted defect "
        "type and an optional Grad-CAM overlay highlighting the influential region."
    )

    uploaded = st.file_uploader(
        "Product image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed"
    )
    explain = st.toggle("Show Grad-CAM overlay", value=True)
    use_api = st.checkbox("Send to the API (logs the inspection)", value=True)

    if uploaded is not None:
        data = uploaded.getvalue()
        with st.spinner("Inspecting…"):
            result = api_predict(api_url, data, uploaded.name, explain=explain) if use_api else None
            source = "API"
            if result is None:
                result = local_inspect(data, uploaded.name, explain=explain)
                source = "local model"
            if result is None:
                st.error(
                    "No model available. Train one with `python scripts/train.py`, or start the API."
                )
                return

        overlay = None
        if result.get("heatmap_png_base64"):
            overlay = Image.open(BytesIO(base64.b64decode(result["heatmap_png_base64"])))

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Input")
            st.image(data, width="stretch")
        with right:
            st.subheader("Grad-CAM" if overlay else "Result")
            if overlay:
                st.image(overlay, width="stretch")
            else:
                st.info("Enable the Grad-CAM toggle to see where the model looked.")

        defective = result["prediction"] == "defective"
        cols = st.columns(4)
        cols[0].metric("Decision", "DEFECTIVE" if defective else "NORMAL")
        cols[1].metric("Defect type", result["defect_type"])
        cols[2].metric("Confidence", f"{result['confidence']:.1%}")
        cols[3].metric("Latency", f"{result['latency_ms']:.1f} ms")

        st.caption(
            f"Model `{result['model_version']}` · device {result['device']} · source: {source} · "
            f"defective probability {result['defective_probability']:.3f}"
        )
        if defective:
            st.warning("Manual inspection recommended before rejecting the unit.")
        else:
            st.success("No defect detected above the decision threshold.")

        probabilities = pd.Series(result["probabilities"], name="probability").sort_values()
        st.bar_chart(probabilities, horizontal=True)
        st.caption(
            "Softmax confidence is not a calibrated probability; the 0.5 defect threshold is "
            "configurable and should be tuned against the cost of misses."
        )
    else:
        st.info("Upload an image to start an inspection.")

    st.subheader("Recent inspections")
    records = api_get(api_url, "/predictions", params={"limit": 10}) if use_api else None
    if records is None:
        repository = Repository(settings=SETTINGS)
        try:
            records = repository.recent(limit=10)
        finally:
            repository.close()
    if records:
        frame = pd.DataFrame(records)
        columns = [
            "created_at",
            "filename",
            "prediction",
            "defect_type",
            "confidence",
            "latency_ms",
            "status",
        ]
        st.dataframe(frame[[c for c in columns if c in frame]], width="stretch", hide_index=True)
    else:
        st.caption("No inspections logged yet.")


# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
def render_model() -> None:
    st.header("Model performance")
    report = evaluation_report()
    if not report:
        st.info("No evaluation report yet. Run `python scripts/evaluate.py`.")
        return

    binary = report["binary"]
    multi = report["multiclass"]

    st.caption(
        f"Model `{report['model_version']}` · split {report['split']} "
        f"({report['n_images']} images) · device {report['latency']['device']}"
    )
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{binary['accuracy']:.3f}")
    cols[1].metric("Precision", f"{binary['precision']:.3f}")
    cols[2].metric("Recall", f"{binary['recall']:.3f}")
    cols[3].metric("F1", f"{binary['f1']:.3f}")
    cols[4].metric("ROC AUC", f"{binary['roc_auc']}" if binary["roc_auc"] else "n/a")
    cols[5].metric("Macro F1", f"{multi['macro_f1']:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Binary confusion matrix")
        matrix = binary["confusion_matrix"]
        table = pd.DataFrame(
            [[matrix["true_negative"], matrix["false_positive"]],
             [matrix["false_negative"], matrix["true_positive"]]],
            index=["Actual normal", "Actual defective"],
            columns=["Predicted normal", "Predicted defective"],
        )
        st.dataframe(table, width="stretch")
        st.caption(
            f"False negatives: {matrix['false_negative']} · false positives: "
            f"{matrix['false_positive']} · threshold {binary['threshold']:.2f}"
        )

        st.subheader("Per-class performance")
        st.dataframe(pd.DataFrame(multi["per_class"]), width="stretch", hide_index=True)

    with right:
        st.subheader("Training curves")
        history = report.get("training") or training_history()
        if history:
            frame = pd.DataFrame(history)
            st.line_chart(frame, x="epoch", y=["train_loss", "val_loss"], height=220)
            st.line_chart(frame, x="epoch", y=["train_accuracy", "val_accuracy"], height=200)
        else:
            st.info("No training history found.")

        benchmark = benchmark_report()
        if benchmark:
            st.subheader("Latency benchmark")
            st.dataframe(pd.DataFrame(benchmark["batches"]), width="stretch", hide_index=True)
            st.caption(f"Single image end-to-end: {benchmark['single_image_ms']} ms")

    with st.expander("Model card"):
        st.json(model_card() or {})


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
def render_monitoring() -> None:
    st.header("Production monitoring")
    summary = api_get(api_url, "/monitoring/summary")
    origin = "API"
    if summary is None:
        repository = Repository(settings=SETTINGS)
        try:
            summary = repository.summary()
        finally:
            repository.close()
        origin = "local database"

    if not summary or summary.get("total_inspections", 0) == 0:
        st.info("No inspections yet. Run `python scripts/simulate_inspections.py --n 50`.")
        return

    st.caption(
        f"Source: {origin} · last inspection {summary.get('last_inspection_at')}"
    )
    cols = st.columns(6)
    cols[0].metric("Inspections", f"{summary['total_inspections']:,}")
    cols[1].metric("Normal", f"{summary['normal_count']:,}")
    cols[2].metric("Defective", f"{summary['defective_count']:,}")
    cols[3].metric("Defect rate", f"{(summary['defect_rate'] or 0):.1%}")
    cols[4].metric("Avg latency", f"{summary['avg_latency_ms'] or 0:.1f} ms")
    cols[5].metric("Failures", f"{summary['failure_count']:,}")

    left, right = st.columns(2)
    with left:
        st.subheader("Defect types")
        defect_types = summary.get("defect_types") or {}
        if defect_types:
            st.bar_chart(pd.Series(defect_types, name="inspections"), horizontal=True)
        else:
            st.info("No defects recorded yet.")
    with right:
        st.subheader("Daily volume")
        daily = pd.DataFrame(summary.get("daily", []))
        if not daily.empty:
            st.line_chart(daily, x="date", y="inspections", height=260)
            st.line_chart(daily, x="date", y="avg_latency_ms", height=200)
        else:
            st.info("No daily history yet.")

    st.subheader("Recent inspections")
    records = api_get(api_url, "/predictions", params={"limit": 50})
    if records is None:
        repository = Repository(settings=SETTINGS)
        try:
            records = repository.recent(limit=50)
        finally:
            repository.close()
    st.dataframe(pd.DataFrame(records), width="stretch", height=320, hide_index=True)
    st.caption(
        "Drift is not measured yet: collect human-verified outcomes and compare them against "
        "these predictions after the observation window."
    )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def render_dataset() -> None:
    st.header("Dataset")
    info, distribution, balance = dataset_overview()
    if info is None:
        st.error(
            f"Dataset not found at `{SETTINGS.dataset_dir}`. "
            "Run `python scripts/generate_demo_dataset.py`."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Images", f"{info['total']:,}")
    cols[1].metric("Classes", info["num_classes"])
    cols[2].metric("Train", info["split_counts"]["train"].__len__() and sum(info["split_counts"]["train"].values()))
    cols[3].metric("Test", sum(info["split_counts"]["test"].values()))

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Class distribution")
        st.dataframe(pd.DataFrame(distribution), width="stretch", hide_index=True)
    with right:
        st.subheader("Sample images (test split)")
        preview = dataset_preview()
        if preview:
            st.image(preview, width="stretch")

    st.subheader("Balance & leakage checks")
    check = split_check()
    if check:
        if check["ok"]:
            st.success(
                f"{check['n_files']} unique images, no duplicates across train/val/test."
            )
        else:
            st.error(
                f"{len(check['duplicates'])} duplicate hashes across splits and "
                f"{len(check['missing'])} missing files."
            )
    else:
        st.info("No `splits.json` manifest found.")
    st.json(balance)


PAGES = {
    "Inspect": render_inspect,
    "Model performance": render_model,
    "Monitoring": render_monitoring,
    "Dataset": render_dataset,
}

PAGES[page]()

"""FastAPI inspection service (Feature 7).

Endpoints
---------
``GET  /health``              service, model and database status
``POST /predict``             inspect one image (optional Grad-CAM overlay)
``POST /predict/batch``       inspect several images in one forward pass
``GET  /model-info``          architecture, classes, threshold, metrics
``GET  /predictions``         inspection history
``GET  /monitoring/summary``  counts, latency, failures, defect mix
``GET  /evaluation/latest``   last test-set report
``POST /evaluation/run``      evaluate the saved model on the test split
"""

from __future__ import annotations

import base64
import hashlib
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from defectiq import __version__
from defectiq.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringSummary,
    PredictionOut,
    PredictionRecordOut,
)
from defectiq.config import Settings, get_settings, resolve_device
from defectiq.db.repository import Repository
from defectiq.inference.predictor import DefectPredictor, Prediction
from defectiq.logging_config import get_logger
from defectiq.modeling.evaluate import (
    evaluate_test_split,
    load_report,
    save_report,
    write_model_card,
)
from defectiq.preprocessing.transforms import ImageValidationError
from defectiq.services import build_services

logger = get_logger(__name__)

DESCRIPTION = """
Upload a product image and get a normal/defective decision, the predicted defect
type, a confidence score and an optional Grad-CAM heatmap.

Every inspection is stored with its latency, model version and device, so the
monitoring endpoints can report rejection rates, throughput and failures.
"""

MAX_BATCH_FILES = 16


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = build_services(cfg)
        app.state.settings = cfg
        app.state.services = services
        app.state.started_at = datetime.now(timezone.utc).isoformat()

        if services.predictor is not None:
            logger.info(
                "DefectIQ ready | model=%s | device=%s",
                services.predictor.version,
                services.predictor.device,
            )
        else:
            logger.warning("DefectIQ started without a model: %s", services.model_error)

        try:
            yield
        finally:
            services.close()

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_process_time(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _services(request: Request):
        services = getattr(request.app.state, "services", None)
        if services is None:  # pragma: no cover - lifespan always sets it
            raise HTTPException(status_code=503, detail="Service container is not ready.")
        return services

    def _predictor(request: Request) -> DefectPredictor:
        services = _services(request)
        if services.predictor is None:
            detail = services.model_error or "No trained model is available."
            raise HTTPException(
                status_code=503,
                detail=f"{detail} Train one with `python scripts/train.py`.",
            )
        return services.predictor

    def _repository(request: Request) -> Repository:
        return _services(request).repository

    def _read_upload(file: UploadFile) -> tuple[bytes, str]:
        data = file.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(data) > cfg.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is larger than {cfg.max_upload_mb} MB.",
            )
        return data, (file.filename or "upload.jpg")[:255]

    def _log(
        request: Request,
        *,
        filename: str,
        data: bytes,
        prediction: Prediction,
        status: str = "success",
        error: str | None = None,
    ) -> dict[str, Any]:
        return _repository(request).log_prediction(
            filename=filename,
            image_sha256=hashlib.sha256(data).hexdigest(),
            content_type="image/*",
            size_bytes=len(data),
            image_width=prediction.image_width,
            image_height=prediction.image_height,
            prediction=prediction.prediction,
            defect_type=prediction.defect_type,
            confidence=prediction.confidence,
            defective_probability=prediction.defective_probability,
            model_name=prediction.model_name,
            model_version=prediction.model_version,
            latency_ms=prediction.latency_ms,
            device=prediction.device,
            status=status,
            error=error,
        )

    def _record_failure(
        request: Request, *, filename: str, data: bytes, message: str
    ) -> None:
        """Persist failed inspections so the failure rate is observable."""
        services = _services(request)
        predictor = services.predictor
        try:
            services.repository.log_prediction(
                filename=filename,
                image_sha256=hashlib.sha256(data).hexdigest() if data else "",
                size_bytes=len(data),
                prediction="normal",
                defect_type="none",
                confidence=0.0,
                defective_probability=0.0,
                model_name=predictor.model_name if predictor else "unknown",
                model_version=predictor.version if predictor else "unknown",
                latency_ms=0.0,
                device=predictor.device if predictor else "cpu",
                status="failed",
                error=message[:500],
            )
        except Exception:  # noqa: BLE001 - logging must never mask the real error
            logger.exception("Could not persist failed inspection")

    # ------------------------------------------------------------------ #
    # Meta
    # ------------------------------------------------------------------ #
    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "name": "DefectIQ",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(request: Request) -> HealthResponse:
        services = _services(request)
        predictor = services.predictor
        database = "ok"
        total = 0
        try:
            total = services.repository.count()
        except Exception:  # noqa: BLE001
            database = "unavailable"

        healthy = predictor is not None and database == "ok"
        return HealthResponse(
            status="ok" if healthy else "degraded",
            model_loaded=predictor is not None,
            model_name=predictor.model_name if predictor else None,
            model_version=predictor.version if predictor else None,
            device=predictor.device if predictor else resolve_device(cfg.device),
            database=database,
            total_inspections=total,
            version=__version__,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.get("/model-info", response_model=ModelInfoResponse, tags=["model"])
    def model_info(request: Request) -> ModelInfoResponse:
        return ModelInfoResponse(**_predictor(request).info())

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    @app.post("/predict", response_model=PredictionOut, tags=["prediction"])
    def predict(
        request: Request,
        file: UploadFile = File(..., description="Product image (JPEG/PNG/BMP/WEBP)"),
        explain: bool = Query(default=False, description="Include a Grad-CAM overlay (base64 PNG)"),
    ) -> PredictionOut:
        predictor = _predictor(request)
        data, filename = _read_upload(file)

        heatmap: str | None = None
        started = time.perf_counter()
        try:
            if explain:
                prediction, png = predictor.explain(data)
                heatmap = base64.b64encode(png).decode("ascii")
            else:
                prediction = predictor.predict_bytes(data)
        except ImageValidationError as exc:
            _record_failure(request, filename=filename, data=data, message=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - inference failures are logged, not hidden
            logger.exception("Inference failed")
            _record_failure(request, filename=filename, data=data, message=str(exc))
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        record = _log(request, filename=filename, data=data, prediction=prediction)
        logger.info(
            "Inspected %s: %s (%s) in %.1f ms",
            filename,
            prediction.prediction,
            prediction.defect_type,
            elapsed_ms,
        )
        return PredictionOut(
            **prediction.to_dict(),
            id=record["id"],
            created_at=record["created_at"],
            filename=filename,
            heatmap_png_base64=heatmap,
        )

    @app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
    def predict_batch(
        request: Request,
        files: list[UploadFile] = File(..., description="Up to 16 product images"),
    ) -> BatchPredictionResponse:
        predictor = _predictor(request)
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded.")
        if len(files) > MAX_BATCH_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"At most {MAX_BATCH_FILES} images per batch.",
            )

        payloads = [_read_upload(file) for file in files]
        started = time.perf_counter()
        try:
            predictions = predictor.predict_batch([data for data, _ in payloads])
        except ImageValidationError as exc:
            for data, filename in payloads:
                _record_failure(request, filename=filename, data=data, message=str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results: list[PredictionOut] = []
        for prediction, (data, filename) in zip(predictions, payloads, strict=True):
            record = _log(request, filename=filename, data=data, prediction=prediction)
            results.append(
                PredictionOut(
                    **prediction.to_dict(),
                    id=record["id"],
                    created_at=record["created_at"],
                    filename=filename,
                )
            )

        return BatchPredictionResponse(
            count=len(results),
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            results=results,
        )

    # ------------------------------------------------------------------ #
    # History & monitoring
    # ------------------------------------------------------------------ #
    @app.get("/predictions", response_model=list[PredictionRecordOut], tags=["history"])
    def predictions(
        request: Request,
        limit: int = Query(default=50, ge=1, le=500),
        prediction: str | None = Query(default=None, pattern="^(normal|defective)$"),
    ) -> list[PredictionRecordOut]:
        rows = _repository(request).recent(limit=limit, prediction=prediction)
        return [PredictionRecordOut(**row) for row in rows]

    @app.get("/monitoring/summary", response_model=MonitoringSummary, tags=["monitoring"])
    def monitoring_summary(request: Request) -> MonitoringSummary:
        return MonitoringSummary(**_repository(request).summary())

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    @app.get("/evaluation/latest", tags=["evaluation"])
    def evaluation_latest(request: Request) -> dict[str, Any]:
        report = load_report(cfg.evaluation_path)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail="No evaluation report yet. Run `POST /evaluation/run` or scripts/evaluate.py.",
            )
        return report

    @app.post("/evaluation/run", tags=["evaluation"])
    def evaluation_run(
        request: Request,
        split: str = Query(default="test", pattern="^(val|test)$"),
    ) -> dict[str, Any]:
        _predictor(request)
        report = evaluate_test_split(cfg, split=split)
        save_report(report, cfg.evaluation_path)
        write_model_card(report, cfg.model_card_path)
        return report

    return app


app = create_app()


__all__ = ["app", "create_app"]

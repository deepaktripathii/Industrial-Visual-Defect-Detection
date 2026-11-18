# API reference

Base URL `http://localhost:8000` · interactive docs `/docs`.

## `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "resnet18",
  "model_version": "resnet18-2026.09.18-104233",
  "device": "cpu",
  "database": "ok",
  "total_inspections": 63,
  "version": "1.0.0",
  "timestamp": "2026-09-18T10:42:35+00:00"
}
```

`degraded` means the model is missing or the database is unreachable; `/predict`
returns 503 in that case.

## `POST /predict?explain=false`

Multipart form field `file`. JPEG, PNG, BMP or WEBP, max 10 MB.

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/demo_dataset/test/scratch/scratch_000.jpg"
```

```json
{
  "id": 64,
  "created_at": "2026-09-18T10:43:02.114Z",
  "filename": "scratch_000.jpg",
  "prediction": "defective",
  "defect_type": "scratch",
  "confidence": 0.9338,
  "defective_probability": 0.9972,
  "probabilities": {
    "crack": 0.0592,
    "dent": 0.0023,
    "good": 0.0028,
    "scratch": 0.9338,
    "stain": 0.002
  },
  "model_name": "resnet18",
  "model_version": "resnet18-2026.09.18-104233",
  "image_size": 224,
  "device": "cpu",
  "latency_ms": 37.7,
  "image_width": 224,
  "image_height": 224,
  "heatmap_png_base64": null
}
```

With `?explain=true` the response also contains `heatmap_png_base64`: a PNG of
the JET heatmap blended over the uploaded image.

Errors:

| Status | Meaning |
| --- | --- |
| 400 | Empty file, not a decodable image |
| 413 | File larger than `DEFECT_MAX_UPLOAD_MB` |
| 500 | Inference failure (logged with `status=failed`) |
| 503 | No checkpoint loaded |

## `POST /predict/batch`

Form field `files` repeated up to 16 times. One forward pass is executed for the
whole batch.

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@good_000.jpg" \
  -F "files=@dent_000.jpg"
```

```json
{
  "count": 2,
  "latency_ms": 41.3,
  "results": [{ "prediction": "normal", "...": "..." }, { "prediction": "defective", "...": "..." }]
}
```

## `GET /model-info`

```json
{
  "model_name": "resnet18",
  "version": "resnet18-2026.09.18-104233",
  "architecture": "resnet18",
  "trained_at": "2026-09-18T10:42:33+00:00",
  "classes": ["crack", "dent", "good", "scratch", "stain"],
  "image_size": 224,
  "threshold": 0.5,
  "device": "cpu",
  "model_path": "artifacts/models/resnet18_defect.pt",
  "model_size_mb": 42.72,
  "metrics": { "val_accuracy": 1.0, "val_f1_macro": 1.0, "val_loss": 0.1047 },
  "epoch": 7
}
```

## `GET /predictions?limit=50&prediction=defective`

Inspection history, newest first. Each row includes the file hash, prediction,
defect type, confidence, defective probability, model version, latency, device
and status (`success` or `failed` with an error message).

## `GET /monitoring/summary`

```json
{
  "total_inspections": 63,
  "window_size": 63,
  "normal_count": 17,
  "defective_count": 46,
  "failure_count": 0,
  "defect_rate": 0.7302,
  "defect_types": { "crack": 8, "dent": 10, "scratch": 14, "stain": 9 },
  "avg_confidence": 0.9341,
  "avg_latency_ms": 24.2,
  "p95_latency_ms": 34.1,
  "devices": { "cpu": 63 },
  "model_versions": { "resnet18-2026.09.18-104233": 63 },
  "last_24h": 63,
  "daily": [{ "date": "2026-09-18", "inspections": 63, "avg_latency_ms": 24.2 }],
  "last_inspection_at": "2026-09-18T10:43:02+00:00"
}
```

## `GET /evaluation/latest`

The full test-set report (binary + per-class metrics, confusion matrix, latency,
training curves). `404` before the first evaluation. See
[results.md](results.md) for the current numbers.

## `POST /evaluation/run?split=test`

Re-evaluates the saved checkpoint on `val` or `test`, rewrites
`artifacts/reports/evaluation.json` and `model_card.json`, and returns the report.

## `GET /`

Service name, version and links to `/docs` and `/health`.

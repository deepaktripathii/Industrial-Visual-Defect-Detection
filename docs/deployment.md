# Deployment notes

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DEFECTIQ_DATASET_DIR` | `./data/demo_dataset` | train/val/test folders |
| `DEFECTIQ_ARTIFACT_DIR` | `./artifacts` | checkpoints, reports, benchmark |
| `DEFECTIQ_DATABASE_URL` | SQLite in `data/` | inspection log |
| `DEFECTIQ_IMAGE_SIZE` | `224` | must match the checkpoint |
| `DEFECTIQ_DEFECTIVE_THRESHOLD` | `0.5` | binary accept/reject cut-off |
| `DEFECTIQ_DEVICE` | `auto` | `cpu`, `cuda`, `mps` or `auto` |
| `DEFECTIQ_BATCH_SIZE` | `32` | evaluation batch size |
| `DEFECTIQ_MAX_UPLOAD_MB` | `10` | upload guard |

## Local

```bash
pip install -r requirements.txt && pip install -e .
python scripts/generate_demo_dataset.py
python scripts/train.py
uvicorn defectiq.api.main:app --reload
streamlit run src/defectiq/dashboard/app.py
```

## Docker Compose

```bash
docker compose up --build
# API       -> http://localhost:8000/docs
# Dashboard -> http://localhost:8501
```

The image installs CPU PyTorch, bakes the ImageNet weights, and the entrypoint
trains a baseline model if `artifacts/models/resnet18_defect.pt` is missing. The
`artifacts/` volume keeps the checkpoint between restarts; set
`DEFECTIQ_BOOTSTRAP_MODEL=0` to start without training.

## GPU

Build a CUDA image (matching the host driver) and expose the GPU:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
```

```yaml
services:
  api:
    build: .
    environment:
      DEFECTIQ_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

`resolve_device` reports the active device in `/health`, `/model-info` and every
stored prediction, so mixed fleets are visible in monitoring.

## Throughput

- Use `/predict/batch` for line integration: one forward pass for up to 16
  images, ~40% more throughput than one-at-a-time on CPU.
- Benchmark before buying hardware: `python scripts/benchmark.py --batches 1 4 8 16`.
- The current model holds ~45 MB of weights. For edge cameras, export to ONNX or
  TorchScript and consider MobileNetV3/EfficientNet-Lite.

## Threshold and risk

The binary decision is `defective_probability = 1 - P(good) >= threshold`.
Raising the threshold reduces false alarms and increases missed defects. Choose
it from the cost ratio (cost of a shipped defect vs cost of a manual re-check)
using the validation ROC curve, then freeze it in the checkpoint metadata.

## Monitoring

`GET /monitoring/summary` reports volume, normal/defective counts, defect-type
mix, average and p95 latency, devices, model versions and failed inspections.
Two operational rules of thumb:

- A rising failure rate usually means a camera, upload format or decoder change,
  not a model problem.
- A sudden shift in defect-type mix or a drop in average confidence is the first
  signal of input drift, long before labelled accuracy is available.

True accuracy drift needs human-verified outcomes. Store them next to the
prediction rows (same schema, add a `verified_label`) and compare weekly.

## Retraining and rollback

1. Train a new checkpoint (`python scripts/train.py`) - it gets a timestamped
   version.
2. Evaluate on the fixed test split (`python scripts/evaluate.py`) and compare
   with `docs/results.md`.
3. Keep the previous checkpoint; roll back by restoring the `.pt` file and
   restarting the API. The prediction log records which version scored each
   image, so incident analysis stays possible after a rollback.

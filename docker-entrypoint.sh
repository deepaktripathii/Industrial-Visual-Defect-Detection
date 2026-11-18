#!/bin/sh
# Prepare the container: directories and, on first boot, a trained checkpoint.
set -eu

: "${DEFECT_DATA_DIR:=/app/data}"
: "${DEFECT_ARTIFACT_DIR:=/app/artifacts}"
: "${DEFECT_BOOTSTRAP_MODEL:=1}"
export DEFECT_DATA_DIR DEFECT_ARTIFACT_DIR

mkdir -p "$DEFECT_DATA_DIR/uploads" "$DEFECT_ARTIFACT_DIR/models" "$DEFECT_ARTIFACT_DIR/reports"

MODEL_PATH="$DEFECT_ARTIFACT_DIR/models/resnet18_defect.pt"
if [ "$DEFECT_BOOTSTRAP_MODEL" = "1" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] no checkpoint found, training a baseline model"
    python scripts/train.py --artifacts "$DEFECT_ARTIFACT_DIR" || true
fi

exec "$@"

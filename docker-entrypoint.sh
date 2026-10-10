#!/bin/sh
# Prepare the container: directories and, on first boot, a trained checkpoint.
set -eu

: "${DEFECTIQ_DATA_DIR:=/app/data}"
: "${DEFECTIQ_ARTIFACT_DIR:=/app/artifacts}"
: "${DEFECTIQ_BOOTSTRAP_MODEL:=1}"
export DEFECTIQ_DATA_DIR DEFECTIQ_ARTIFACT_DIR

mkdir -p "$DEFECTIQ_DATA_DIR/uploads" "$DEFECTIQ_ARTIFACT_DIR/models" "$DEFECTIQ_ARTIFACT_DIR/reports"

MODEL_PATH="$DEFECTIQ_ARTIFACT_DIR/models/resnet18_defect.pt"
if [ "$DEFECTIQ_BOOTSTRAP_MODEL" = "1" ] && [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] no checkpoint found, training a baseline model"
    python scripts/train.py --artifacts "$DEFECTIQ_ARTIFACT_DIR" || true
fi

exec "$@"

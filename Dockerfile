# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    DEFECTIQ_DATA_DIR=/app/data \
    DEFECTIQ_ARTIFACT_DIR=/app/artifacts \
    DEFECTIQ_DEVICE=cpu

WORKDIR /app

# libglib2.0 is required by the OpenCV wheel on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels keep the image smaller than the default CUDA build.
RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Bake the ImageNet weights so first-boot training works offline.
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)"

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY data/demo_dataset ./data/demo_dataset
COPY data/README.md ./data/README.md
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN pip install --no-deps -e . && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "defectiq.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8000

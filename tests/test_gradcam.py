"""Feature 6: Grad-CAM heatmaps and overlays."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection.explain.gradcam import GradCAM, overlay_heatmap, overlay_png_bytes
from defect_detection.modeling.model import load_checkpoint


def test_gradcam_returns_normalised_heatmap(trained_settings):
    model, _payload = load_checkpoint(trained_settings.model_path, device="cpu")
    gradcam = GradCAM(model, model.layer4[-1])

    try:
        cam = gradcam.generate(torch.randn(1, 3, 64, 64), out_size=(64, 64))
    finally:
        gradcam.remove()

    assert cam.shape == (64, 64)
    assert np.isfinite(cam).all()
    assert 0.0 <= float(cam.min()) <= float(cam.max()) <= 1.0


def test_gradcam_rejects_bad_batch(trained_settings):
    model, _payload = load_checkpoint(trained_settings.model_path, device="cpu")
    gradcam = GradCAM(model, model.layer4[-1])
    try:
        with pytest.raises(ValueError):
            gradcam.generate(torch.randn(2, 3, 64, 64))
    finally:
        gradcam.remove()


def test_overlay_matches_image_size():
    image = Image.new("RGB", (80, 60), (150, 150, 150))
    cam = np.linspace(0, 1, 60 * 60, dtype=np.float32).reshape(60, 60)

    overlay = overlay_heatmap(image, cam)

    assert overlay.shape == (60, 80, 3)
    assert overlay.dtype == np.uint8


def test_overlay_png_bytes():
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    cam = np.zeros((32, 32), dtype=np.float32)

    png = overlay_png_bytes(image, cam)

    assert png.startswith(b"\x89PNG")
    assert len(png) > 100

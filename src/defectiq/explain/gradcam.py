"""Grad-CAM (Feature 6).

Grad-CAM weights the final convolutional feature maps by the gradient of the
predicted class score. The result is a coarse heatmap of the regions that pushed
the decision - useful for spotting whether the network looked at the scratch or
at the background.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from defectiq.preprocessing.transforms import open_image


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inputs, output) -> None:
        self._activations = output

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def generate(
        self,
        tensor: torch.Tensor,
        *,
        class_index: int | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a normalised heatmap for one image tensor ``(1, 3, H, W)``."""
        if tensor.dim() != 4 or tensor.size(0) != 1:
            raise ValueError("Grad-CAM expects a single image tensor shaped (1, 3, H, W).")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        tensor = tensor.detach().clone().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(tensor)
            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())
            score = logits[0, class_index]
            score.backward()

        if self._activations is None or self._gradients is None:  # pragma: no cover
            raise RuntimeError("Grad-CAM hooks did not capture activations.")

        activations = self._activations.detach()[0]
        gradients = self._gradients.detach()[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        cam = cam - cam.min()
        denominator = cam.max().clamp_min(1e-8)
        cam = cam / denominator

        height, width = out_size or (tensor.size(2), tensor.size(3))
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        )
        self.model.zero_grad(set_to_none=True)
        return cam.squeeze().cpu().numpy().astype(np.float32)

    def remove(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(
    image: Image.Image | np.ndarray | str | Path,
    cam: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a JET heatmap over the original image and return RGB uint8."""
    import cv2

    if not isinstance(image, np.ndarray):
        image = np.asarray(open_image(image).convert("RGB"))

    if image.shape[:3] != cam.shape[:2] and cam.shape[:2] != image.shape[:2]:
        cam = cv2.resize(cam, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    heat = cv2.applyColorMap(np.uint8(np.clip(cam, 0, 1) * 255), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, heat, alpha, 0)


def overlay_png_bytes(
    image: Image.Image | np.ndarray | str | Path,
    cam: np.ndarray,
    *,
    alpha: float = 0.45,
) -> bytes:
    """Encode the overlay as PNG for API responses."""
    import cv2

    overlay = overlay_heatmap(image, cam, alpha=alpha)
    success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:  # pragma: no cover - cv2 failure is exceptional
        raise RuntimeError("Could not encode the Grad-CAM overlay.")
    return buffer.tobytes()


__all__ = ["GradCAM", "overlay_heatmap", "overlay_png_bytes"]

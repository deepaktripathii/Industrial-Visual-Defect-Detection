"""Image preprocessing and augmentation (Feature 2).

Augmentation is applied *only* to the training split. Validation and test use a
deterministic resize + centre crop so metrics are comparable between runs.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from defectiq.config import IMAGENET_MEAN, IMAGENET_STD


class ImageValidationError(ValueError):
    """The uploaded file is not a decodable image."""


def build_train_transform(image_size: int = 224, *, augment: bool = True):
    """Training transform: light, label-preserving augmentation for surfaces."""
    if not augment:
        return build_eval_transform(image_size)

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8, fill=128),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224):
    """Deterministic transform for validation, test and inference."""
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def open_image(source: bytes | str | Path | Image.Image) -> Image.Image:
    """Decode, orient and convert any image source to RGB."""
    try:
        if isinstance(source, Image.Image):
            image = source
        elif isinstance(source, (bytes, bytearray)):
            with Image.open(BytesIO(source)) as probe:
                probe.verify()
            image = Image.open(BytesIO(source))
        else:
            with Image.open(source) as probe:
                probe.verify()
            image = Image.open(source)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(f"Could not decode the image: {exc}") from exc

    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def denormalize_tensor(tensor, *, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> np.ndarray:
    """``(3, H, W)`` normalised tensor -> ``(H, W, 3)`` uint8 RGB array."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std) + np.asarray(mean)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


__all__ = [
    "ImageValidationError",
    "build_eval_transform",
    "build_train_transform",
    "denormalize_tensor",
    "image_to_array",
    "open_image",
]

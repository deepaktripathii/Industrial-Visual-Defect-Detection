"""Feature 2: preprocessing, augmentation and image validation."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from defectiq.preprocessing.transforms import (
    ImageValidationError,
    build_eval_transform,
    build_train_transform,
    denormalize_tensor,
    image_to_array,
    open_image,
)


def test_eval_transform_shape_and_dtype():
    transform = build_eval_transform(64)
    tensor = transform(Image.new("RGB", (120, 90), (30, 30, 30)))

    assert tuple(tensor.shape) == (3, 64, 64)
    assert tensor.dtype.is_floating_point
    assert float(tensor.min()) < 0  # normalisation pushes dark pixels below zero


def test_train_transform_output_is_normalised():
    transform = build_train_transform(64)
    tensor = transform(Image.new("RGB", (80, 80), (200, 200, 200)))

    assert tuple(tensor.shape) == (3, 64, 64)
    assert float(tensor.max()) > 1  # ImageNet normalisation


def test_augmentation_can_be_disabled():
    transform = build_train_transform(64, augment=False)
    tensor = transform(Image.new("RGB", (90, 90), (120, 120, 120)))
    assert tuple(tensor.shape) == (3, 64, 64)


def test_open_image_from_bytes():
    from io import BytesIO

    buffer = BytesIO()
    Image.new("RGB", (32, 32), (10, 20, 30)).save(buffer, format="PNG")
    image = open_image(buffer.getvalue())

    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_open_image_from_path(tiny_dataset):
    path = next((tiny_dataset / "test" / "good").glob("*.jpg"))
    image = open_image(path)

    assert image.mode == "RGB"


def test_invalid_bytes_raise():
    with pytest.raises(ImageValidationError):
        open_image(b"this is definitely not an image")


def test_denormalize_restores_uint8():
    transform = build_eval_transform(64)
    tensor = transform(Image.new("RGB", (64, 64), (200, 120, 40)))
    array = denormalize_tensor(tensor)

    assert array.shape == (64, 64, 3)
    assert array.dtype == np.uint8
    assert 180 <= array[0, 0, 0] <= 220


def test_image_to_array():
    array = image_to_array(Image.new("RGB", (10, 5), (1, 2, 3)))

    assert array.shape == (5, 10, 3)
    assert array[0, 0].tolist() == [1, 2, 3]

"""Feature 3: model construction, freezing and checkpoint round trip."""

from __future__ import annotations

import torch

from defect_detection.modeling.model import (
    build_model,
    load_checkpoint,
    make_version,
    parameter_report,
    save_checkpoint,
)


def test_forward_shape():
    model = build_model(5, pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randn(2, 3, 64, 64))

    assert tuple(logits.shape) == (2, 5)


def test_freeze_backbone_keeps_head_trainable():
    model = build_model(3, pretrained=False, freeze_backbone=True)
    report = parameter_report(model)

    assert report["trainable"] < report["total"]
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_parameter_counts_are_consistent():
    model = build_model(5, pretrained=False, freeze_backbone=False)
    report = parameter_report(model)

    assert report["trainable"] == report["total"]
    assert report["total"] > 11_000_000


def test_checkpoint_roundtrip(tmp_path):
    classes = ["crack", "good", "scratch"]
    model = build_model(len(classes), pretrained=False)
    model.eval()

    path = save_checkpoint(
        tmp_path / "checkpoint.pt",
        model,
        classes=classes,
        image_size=64,
        version="resnet18-test",
        metrics={"val_f1_macro": 0.5},
        threshold=0.4,
    )
    loaded, payload = load_checkpoint(path, device="cpu")
    loaded.eval()

    assert payload["classes"] == classes
    assert payload["version"] == "resnet18-test"
    assert payload["defective_threshold"] == 0.4
    assert payload["image_size"] == 64

    sample = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(sample), loaded(sample), atol=1e-5)


def test_version_format():
    version = make_version()

    assert version.startswith("resnet18-")
    assert len(version.split("-")) >= 3

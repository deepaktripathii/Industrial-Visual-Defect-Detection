"""Feature 3: the training loop produces a checkpoint and history."""

from __future__ import annotations

from pathlib import Path

from defectiq.modeling.trainer import load_history, train_model


def test_training_writes_checkpoint_and_history(tmp_settings, tiny_dataset):
    result = train_model(
        tmp_settings,
        pretrained=False,
        epochs=2,
        batch_size=2,
        freeze_epochs=0,
        patience=0,
        progress=None,
    )

    assert Path(result.model_path).exists()
    assert len(result.history) == 2
    assert result.history[0]["epoch"] == 1
    assert 0.0 <= result.best_val_f1 <= 1.0
    assert result.parameters["total"] > 0
    assert result.total_seconds > 0
    assert result.device == "cpu"

    history = load_history(tmp_settings.history_path)
    assert len(history) == 2
    assert {"train_loss", "val_loss", "train_accuracy", "val_accuracy"} <= set(history[0])


def test_training_rejects_mismatched_splits(tmp_settings, tiny_dataset, tmp_path):
    import shutil

    shutil.copytree(tiny_dataset, tmp_path / "broken")
    shutil.rmtree(tmp_path / "broken" / "val" / "good")
    (tmp_path / "broken" / "val" / "good").mkdir()

    try:
        train_model(
            tmp_settings,
            data_dir=tmp_path / "broken",
            pretrained=False,
            epochs=1,
            batch_size=2,
            freeze_epochs=0,
        )
    except Exception as exc:  # noqa: BLE001 - the exact type is a detail
        assert "class" in str(exc).lower() or "missing" in str(exc).lower()
    else:  # pragma: no cover - defensive
        raise AssertionError("Training should reject an empty class folder")

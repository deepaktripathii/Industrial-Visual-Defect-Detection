"""Runtime configuration for DefectIQ.

Everything is overridable with ``DEFECTIQ_*`` environment variables (see
``.env.example``). One cached ``Settings`` object is shared by the training
scripts, the API, the dashboard and the tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Folder name of the defect-free class in every split.
NORMAL_CLASS = "good"


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    project_root: Path
    data_dir: Path
    dataset_dir: Path
    upload_dir: Path
    artifact_dir: Path
    model_dir: Path
    report_dir: Path

    database_url: str

    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    freeze_epochs: int = 1
    seed: int = 42
    device: str = "auto"

    defective_threshold: float = 0.5
    max_upload_mb: int = 10

    model_name: str = "resnet18"
    app_name: str = "DefectIQ - Industrial Visual Inspection API"
    app_version: str = "1.0.0"
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    # Derived paths
    # ------------------------------------------------------------------ #
    @property
    def model_path(self) -> Path:
        return self.model_dir / f"{self.model_name}_defect.pt"

    @property
    def history_path(self) -> Path:
        return self.report_dir / "history.json"

    @property
    def evaluation_path(self) -> Path:
        return self.report_dir / "evaluation.json"

    @property
    def model_card_path(self) -> Path:
        return self.report_dir / "model_card.json"

    @property
    def splits_manifest_path(self) -> Path:
        return self.dataset_dir / "splits.json"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.artifact_dir, self.model_dir, self.report_dir):
            path.mkdir(parents=True, exist_ok=True)

    def with_storage(self, root: Path | str) -> Settings:
        """Re-point every mutable path (tests, experiments, scripts)."""
        from dataclasses import replace

        base = Path(root).expanduser().resolve()
        return replace(
            self,
            data_dir=base / "data",
            dataset_dir=base / "data" / "demo_dataset",
            upload_dir=base / "data" / "uploads",
            artifact_dir=base / "artifacts",
            model_dir=base / "artifacts" / "models",
            report_dir=base / "artifacts" / "reports",
            database_url=f"sqlite:///{(base / 'data' / 'defectiq.db').as_posix()}",
        )

    def with_artifacts(self, directory: Path | str) -> Settings:
        from dataclasses import replace

        base = Path(directory).expanduser().resolve()
        return replace(
            self,
            artifact_dir=base,
            model_dir=base / "models",
            report_dir=base / "reports",
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = PROJECT_ROOT
    data_dir = _env_path("DEFECTIQ_DATA_DIR", root / "data")
    artifact_dir = _env_path("DEFECTIQ_ARTIFACT_DIR", root / "artifacts")

    return Settings(
        project_root=root,
        data_dir=data_dir,
        dataset_dir=_env_path("DEFECTIQ_DATASET_DIR", data_dir / "demo_dataset"),
        upload_dir=_env_path("DEFECTIQ_UPLOAD_DIR", data_dir / "uploads"),
        artifact_dir=artifact_dir,
        model_dir=_env_path("DEFECTIQ_MODEL_DIR", artifact_dir / "models"),
        report_dir=_env_path("DEFECTIQ_REPORT_DIR", artifact_dir / "reports"),
        database_url=_env_str(
            "DEFECTIQ_DATABASE_URL", f"sqlite:///{(data_dir / 'defectiq.db').as_posix()}"
        ),
        image_size=_env_int("DEFECTIQ_IMAGE_SIZE", 224),
        batch_size=_env_int("DEFECTIQ_BATCH_SIZE", 32),
        num_workers=_env_int("DEFECTIQ_NUM_WORKERS", 0),
        epochs=_env_int("DEFECTIQ_EPOCHS", 8),
        learning_rate=_env_float("DEFECTIQ_LEARNING_RATE", 1e-4),
        weight_decay=_env_float("DEFECTIQ_WEIGHT_DECAY", 1e-4),
        freeze_epochs=_env_int("DEFECTIQ_FREEZE_EPOCHS", 1),
        seed=_env_int("DEFECTIQ_SEED", 42),
        device=_env_str("DEFECTIQ_DEVICE", "auto"),
        defective_threshold=_env_float("DEFECTIQ_DEFECTIVE_THRESHOLD", 0.5),
        max_upload_mb=_env_int("DEFECTIQ_MAX_UPLOAD_MB", 10),
        log_level=_env_str("DEFECTIQ_LOG_LEVEL", "INFO"),
    )


def reset_settings_cache() -> None:
    get_settings.cache_clear()


def resolve_device(preferred: str | None = None) -> str:
    """Return ``cuda``, ``mps`` or ``cpu`` based on availability."""
    choice = (preferred or get_settings().device).lower()
    if choice == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"

    if choice in {"auto", "cuda"} and torch.cuda.is_available():
        return "cuda"
    if choice in {"auto", "mps"} and getattr(torch.backends, "mps", None) is not None:
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "NORMAL_CLASS",
    "Settings",
    "get_settings",
    "reset_settings_cache",
    "resolve_device",
]

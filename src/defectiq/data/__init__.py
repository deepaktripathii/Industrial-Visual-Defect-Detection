"""Data utilities: discovery, splits and analysis."""

from defectiq.data.analysis import (
    balance_report,
    class_distribution,
    image_size_stats,
    sample_paths,
    save_preview_grid,
)
from defectiq.data.dataset import (
    DatasetInfo,
    DatasetLayoutError,
    build_dataloader,
    build_image_folder,
    class_weights,
    discover_dataset,
)
from defectiq.data.splits import (
    compute_manifest,
    load_manifest,
    save_manifest,
    sha256_file,
    verify_manifest,
)

__all__ = [
    "DatasetInfo",
    "DatasetLayoutError",
    "balance_report",
    "build_dataloader",
    "build_image_folder",
    "class_distribution",
    "class_weights",
    "compute_manifest",
    "discover_dataset",
    "image_size_stats",
    "load_manifest",
    "sample_paths",
    "save_manifest",
    "save_preview_grid",
    "sha256_file",
    "verify_manifest",
]

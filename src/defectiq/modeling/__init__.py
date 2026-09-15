"""Model package: architecture, training and evaluation."""

from defectiq.modeling.evaluate import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    defective_probabilities,
    evaluate_test_split,
    load_report,
    predict_split,
    save_report,
    write_model_card,
)
from defectiq.modeling.model import (
    build_model,
    load_checkpoint,
    make_version,
    parameter_report,
    save_checkpoint,
)
from defectiq.modeling.trainer import (
    TrainingResult,
    evaluate_loader,
    load_history,
    save_history,
    set_seed,
    train_model,
    train_one_epoch,
)

__all__ = [
    "TrainingResult",
    "build_model",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "defective_probabilities",
    "evaluate_loader",
    "evaluate_test_split",
    "load_checkpoint",
    "load_history",
    "load_report",
    "make_version",
    "parameter_report",
    "predict_split",
    "save_checkpoint",
    "save_history",
    "save_report",
    "set_seed",
    "train_model",
    "train_one_epoch",
    "write_model_card",
]

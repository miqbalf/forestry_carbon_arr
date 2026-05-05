"""
Machine learning utilities for forestry_carbon_arr.

Currently includes CatBoost-based land-cover training helpers refactored
from exploratory notebooks into reusable functions.
"""

from .catboost_workflow import (  # noqa: F401
    prepare_config,
    train_full_area_model,
    save_config_ml,
    load_model,
    load_full_area_model,
    evaluate_model,
    compute_shap_feature_ranking,
    subset_dataset_features,
    plot_confusion_matrix,
    plot_multiclass_roc,
    save_config_ml,
)
from .vis import plot_unstack_ds  # noqa: F401
from .inference import predict_dataset_features  # noqa: F401

__all__ = [
    "prepare_config",
    "train_full_area_model",
    "save_config_ml",
    "load_model",
    "load_full_area_model",
    "evaluate_model",
    "compute_shap_feature_ranking",
    "subset_dataset_features",
    "plot_confusion_matrix",
    "plot_multiclass_roc",
    "plot_unstack_ds",
    "predict_dataset_features",
]


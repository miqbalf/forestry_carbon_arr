"""
CatBoost training and evaluation workflow refactored from notebooks.

This module exposes small, composable helpers so the CatBoost
classification workflow can be reused outside Jupyter while preserving
the behaviour of the original notebook cells.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

try:  # Plotting is optional; functions guard on availability
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except Exception:  # pragma: no cover - plotting may be unavailable in headless envs
    PLOTTING_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    SHAP_AVAILABLE = False

try:
    import gcsfs

    GCSFS_AVAILABLE = True
except Exception:  # pragma: no cover
    GCSFS_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _hash_config(config: Mapping[str, object]) -> str:
    """Create a stable hash for caching artifacts."""
    return hashlib.md5(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def save_config_ml(
    config: Mapping[str, object],
    local_path: Optional[str] = None,
    gcs_path: Optional[str] = None,
) -> str:
    """
    Persist a config dictionary to disk and optionally upload to GCS.

    Args:
        config: Configuration dictionary (must contain 'hash' if local_path not provided)
        local_path: Optional explicit local path; defaults to cache dir /mnt/data/cache/<hash>/config.json
        gcs_path: Optional GCS URI (gs://bucket/path.json) to upload the saved config

    Returns:
        The local path where the config was saved.
    """
    if local_path is None:
        if "hash" not in config:
            raise ValueError("Config must include 'hash' or specify local_path explicitly.")
        cache_dir = f"/mnt/data/cache/{config['hash']}"
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, "config.json")
    else:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if gcs_path:
        if not GCSFS_AVAILABLE:  # pragma: no cover
            raise ImportError("gcsfs is required to upload config to GCS")
        fs = gcsfs.GCSFileSystem(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            token=os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json"),
        )
        fs.put(local_path, gcs_path)

    return local_path


def _load_json_from_gcs(uri: str) -> Dict[str, object]:
    """Load JSON config from GCS."""
    if not GCSFS_AVAILABLE:  # pragma: no cover
        raise ImportError("gcsfs is required to load config from GCS")
    fs = gcsfs.GCSFileSystem(project=os.getenv("GOOGLE_CLOUD_PROJECT"), token=os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json"))
    with fs.open(uri, "r") as f:
        return json.load(f)


def _copy_gcs_file_to_temp(uri: str, suffix: str = "") -> str:
    """Download a GCS file to a temporary local path and return it."""
    if not GCSFS_AVAILABLE:  # pragma: no cover
        raise ImportError("gcsfs is required to load models from GCS")
    import tempfile

    fs = gcsfs.GCSFileSystem(project=os.getenv("GOOGLE_CLOUD_PROJECT"), token=os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json"))
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with fs.open(uri, "rb") as src, open(tmp_path, "wb") as dst:
        dst.write(src.read())
    return tmp_path


def prepare_config(
    model_config: Mapping[str, object],
    plot_ids: Union[Sequence[Union[str, int]], np.ndarray],
    extra_config: Optional[Mapping[str, object]] = None,
    use_proba: bool = False,
    optimal_threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Prepare a training configuration and cache directory descriptor.

    Parameters mirror the notebook's prepare_config helper but are kept minimal
    to avoid notebook globals.
    """
    plot_ids_list = list(plot_ids.tolist() if isinstance(plot_ids, np.ndarray) else plot_ids)

    config: Dict[str, object] = {
        "model": dict(model_config),
        "plots": plot_ids_list,
        "sample_weight": True,
    }
    if extra_config:
        config.update(extra_config)

    # Build hashable copy excluding noisy fields
    config_for_hash = dict(config)
    model_clean = dict(config_for_hash["model"])
    model_clean.pop("verbose", None)
    config_for_hash["model"] = model_clean
    config_for_hash["hash_meta"] = {"use_proba": use_proba, "optimal_threshold": optimal_threshold}

    config["use_proba"] = use_proba
    config["optimal_threshold"] = optimal_threshold
    config["hash"] = _hash_config(config_for_hash)

    cache_dir = f"/mnt/data/cache/{config['hash']}"
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    return config


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def _extract_features_and_labels(ds) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix and labels from an xarray.Dataset-like object.
    Expects:
        - ds.X_features: 2D (sample, feature)
        - ds.y: 1D (sample,)
    """
    try:
        X = ds.X_features.values
        y = ds.y.values
    except Exception as e:  # pragma: no cover - defensive logging
        raise ValueError("Dataset must expose X_features and y variables") from e
    return X, y


def split_dataset(
    ds,
    validation_split: float = 0.2,
    random_seed: int = 42,
) -> DatasetSplits:
    """Stratified train/validation split preserving sample indices."""
    X, y = _extract_features_and_labels(ds)
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=validation_split,
        random_state=random_seed,
        stratify=y,
    )
    return DatasetSplits(
        X_train=X[train_idx],
        X_val=X[val_idx],
        y_train=y[train_idx],
        y_val=y[val_idx],
        train_indices=train_idx,
        val_indices=val_idx,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_full_area_model(
    ds_full,
    config: Dict[str, object],
    validation_split: float = 0.2,
    random_seed: int = 42,
) -> Tuple[CatBoostClassifier, str]:
    """
    Train a CatBoost classifier on the provided dataset.

    Expects ds_full to have:
        - X_features (sample, feature)
        - y (sample,)
    """
    label_map = config.get("label_map") or {
        int(c): f"Class {int(c)}" for c in sorted(np.unique(ds_full.y.values))
    }
    sorted_classes = sorted(label_map.keys())

    splits = split_dataset(ds_full, validation_split=validation_split, random_seed=random_seed)
    config["train_indices"] = splits.train_indices.tolist()
    config["val_indices"] = splits.val_indices.tolist()

    model = CatBoostClassifier(**config["model"])
    model.fit(
        splits.X_train,
        splits.y_train,
        eval_set=(splits.X_val, splits.y_val),
        verbose=config["model"].get("verbose", 100),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"/mnt/data/cache/{config['hash']}/catboost_{timestamp}.cbm"
    model.save_model(model_path)

    config["model_path"] = model_path
    config["timestamp"] = timestamp
    config["label_map"] = label_map
    config["classes"] = sorted_classes

    logger.info("Model trained and saved to %s", model_path)
    return model, model_path


def load_model(
    config: Optional[Mapping[str, object]] = None,
    timestamp: Optional[str] = None,
    model_path: Optional[str] = None,
    gcs_model_path: Optional[str] = None,
    gcs_config_path: Optional[str] = None,
) -> Tuple[CatBoostClassifier, str, Optional[Dict[str, object]]]:
    """
    Load a CatBoost model.

    Priority for locating the model:
    1. gcs_model_path (download then load)
    2. explicit model_path
    3. config + timestamp (cached hash directory)
    4. latest model in config cache dir

    Config resolution:
    - If gcs_config_path is provided, it will be downloaded and returned.
    - If config is passed, it is returned unchanged (or augmented with local paths).
    - If neither is provided, returns None for config.
    """
    loaded_config: Optional[Dict[str, object]] = None

    if gcs_config_path:
        loaded_config = _load_json_from_gcs(gcs_config_path)
        config = loaded_config

    cache_dir = f"/mnt/data/cache/{config['hash']}" if config and "hash" in config else None

    if gcs_model_path:
        local_model_path = _copy_gcs_file_to_temp(gcs_model_path, suffix=".cbm")
    elif model_path:
        local_model_path = model_path
    elif cache_dir:
        if timestamp:
            local_model_path = os.path.join(cache_dir, f"catboost_{timestamp}.cbm")
        elif config and "model_path" in config:
            local_model_path = config["model_path"]  # type: ignore[assignment]
        else:
            model_files = glob.glob(os.path.join(cache_dir, "catboost_*.cbm"))
            if not model_files:
                raise FileNotFoundError(f"No models found in {cache_dir}")
            local_model_path = max(model_files, key=os.path.getmtime)
    else:
        raise ValueError("Provide at least one of: gcs_model_path, model_path, or config with hash.")

    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Model not found at: {local_model_path}")

    model = CatBoostClassifier()
    model.load_model(local_model_path)
    return model, local_model_path, loaded_config

# Backwards compatibility
def load_full_area_model(config: Mapping[str, object], timestamp: Optional[str] = None) -> Tuple[CatBoostClassifier, str]:
    model, path, _ = load_model(config=config, timestamp=timestamp)
    return model, path


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_model(
    model: CatBoostClassifier,
    ds,
    config: Mapping[str, object],
    reuse_split: bool = True,
) -> Dict[str, object]:
    """
    Evaluate a trained model using stored split indices when available.
    Returns metrics and predictions for downstream plotting.
    """
    X_all, y_all = _extract_features_and_labels(ds)

    if reuse_split and "val_indices" in config and "train_indices" in config:
        val_idx = np.asarray(config["val_indices"])
        train_idx = np.asarray(config["train_indices"])
    else:
        splits = split_dataset(ds)
        val_idx = splits.val_indices
        train_idx = splits.train_indices

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_confusion_matrix": confusion_matrix(y_val, y_val_pred, normalize="true").tolist(),
        "val_classification_report": classification_report(
            y_val,
            y_val_pred,
            labels=sorted(config.get("label_map", {}).keys()) if config.get("label_map") else None,
            target_names=list(config.get("label_map", {}).values()) if config.get("label_map") else None,
            output_dict=True,
        ),
        "val_kappa": float(cohen_kappa_score(y_val, y_val_pred)),
    }

    return {
        "metrics": metrics,
        "y_train": y_train,
        "y_val": y_val,
        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
    }


# --------------------------------------------------------------------------- #
# Plotting helpers (optional)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_label_map: Optional[Mapping[int, str]] = None,
    dataset: str = "validation",
):
    """Render normalized confusion matrix with label mapping."""
    if not PLOTTING_AVAILABLE:  # pragma: no cover
        raise ImportError("matplotlib/seaborn are required for plotting")

    if class_label_map:
        class_names = [class_label_map[int(c)] for c in sorted(class_label_map.keys())]
    else:
        class_names = [f"Class {int(c)}" for c in np.unique(y_true)]
    class_ids = [f"Class {i}" for i in range(len(class_names))]

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > 0.5 else "black",
            )

    ax.set_xticks(range(len(class_ids)))
    ax.set_xticklabels(class_ids, rotation=45, ha="right")
    ax.set_yticks(range(len(class_ids)))
    ax.set_yticklabels(class_ids)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix ({dataset})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    handles = [
        plt.Line2D([0], [0], color="white", marker="s", markersize=10, label=f"{cid} → {name}")
        for cid, name in zip(class_ids, class_names)
    ]
    ax.legend(handles=handles, title="Class Mapping", loc="center left", bbox_to_anchor=(1.25, 0.5), frameon=True)
    plt.tight_layout()
    return fig, ax


def plot_multiclass_roc(
    model: CatBoostClassifier,
    X: np.ndarray,
    y: np.ndarray,
    class_label_map: Optional[Mapping[int, str]] = None,
):
    """Plot multi-class ROC (one-vs-rest)."""
    if not PLOTTING_AVAILABLE:  # pragma: no cover
        raise ImportError("matplotlib/seaborn are required for plotting")

    classes = sorted(class_label_map.keys()) if class_label_map else sorted(np.unique(y))
    y_bin = label_binarize(y, classes=classes)
    y_score = model.predict_proba(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        label = f"Class {cls}"
        if class_label_map:
            label += f" → {class_label_map[cls]} (AUC={roc_auc:.2f})"
        else:
            label += f" (AUC={roc_auc:.2f})"
        ax.plot(fpr, tpr, lw=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multiclass ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------- #
# SHAP-based feature ranking
# --------------------------------------------------------------------------- #
def compute_shap_feature_ranking(
    model: CatBoostClassifier,
    ds,
    top_n: int = 50,
    random_seed: int = 42,
) -> Tuple[List[str], List[float]]:
    """
    Compute mean |SHAP| per feature and return top_n features and scores.
    """
    if not SHAP_AVAILABLE:  # pragma: no cover
        raise ImportError("shap is required for SHAP feature ranking")

    rng = np.random.default_rng(random_seed)
    X, _ = _extract_features_and_labels(ds)
    feature_names = [str(f) for f in ds.coords["feature"].values]

    if len(X) == 0:
        raise ValueError("Dataset is empty; cannot compute SHAP values.")

    sample_size = min(1000, len(X))
    idx = rng.choice(len(X), size=sample_size, replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_array = np.stack([np.abs(sv) for sv in shap_values])
    else:
        shap_array = np.abs(np.asarray(shap_values))

    feature_axis_candidates = [i for i, s in enumerate(shap_array.shape) if s == len(feature_names)]
    if not feature_axis_candidates:
        raise ValueError(f"Could not find feature axis of length {len(feature_names)} in SHAP shape {shap_array.shape}")
    feature_axis = feature_axis_candidates[-1]

    shap_feat_last = np.moveaxis(shap_array, feature_axis, -1)
    mean_abs_shap = shap_feat_last.mean(axis=tuple(range(shap_feat_last.ndim - 1)))

    order = np.argsort(mean_abs_shap)[::-1]
    top_idx = order[:top_n]
    selected_features = [feature_names[i] for i in top_idx]
    selected_importances = [float(mean_abs_shap[i]) for i in top_idx]
    return selected_features, selected_importances


def subset_dataset_features(ds, keep: Sequence[str]):
    """
    Return a view of ds keeping only the provided feature names (matching the
    'feature' coordinate).
    """
    feature_names = list(ds.coords["feature"].values)
    missing = sorted(set(keep) - set(feature_names))
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    idx = [feature_names.index(f) for f in keep]
    return ds.isel(feature=idx)


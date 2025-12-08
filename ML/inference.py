"""
Inference utilities for CatBoost models on large xarray datasets.
"""

from typing import Mapping, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm


def _subset_features(ds, feature_list: Sequence[str], feature_var: str = "X_features"):
    """Subset ds[feature_var] to the provided feature_list order."""
    feats_available = list(ds.coords["feature"].values)
    missing = sorted(set(feature_list) - set(feats_available))
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    idx = [feats_available.index(f) for f in feature_list]
    return ds[feature_var].isel(feature=idx)


def predict_dataset_features(
    model,
    ds,
    config: Mapping[str, object],
    chunk_size: int = 500_000,
    feature_var: str = "X_features",
    coord_x_var: str = "coord_x",
    coord_y_var: str = "coord_y",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference on a large stacked dataset, respecting feature order from config['X_features_list'].

    Args:
        model: Trained CatBoost model with predict().
        ds: xarray Dataset containing feature_var with dims (sample, feature) and coord vars.
        config: Config dict that should include 'X_features_list' for ordering.
        chunk_size: Number of samples per batch.
        feature_var: Name of feature DataArray in ds.
        coord_x_var / coord_y_var: Coordinate variable names for output.

    Returns:
        preds (np.ndarray), coord_x (np.ndarray), coord_y (np.ndarray)
    """
    if "X_features_list" not in config:
        raise ValueError("config missing 'X_features_list' for feature ordering.")

    feature_list = list(config["X_features_list"])
    X_sel = _subset_features(ds, feature_list, feature_var=feature_var)

    n_samples = int(ds.sizes["sample"])
    preds_chunks, cx_chunks, cy_chunks = [], [], []

    for start in tqdm(
        range(0, n_samples, chunk_size),
        desc="Predicting",
        unit="samples",
        total=(n_samples + chunk_size - 1) // chunk_size,
    ):
        end = min(start + chunk_size, n_samples)
        X_chunk = X_sel.isel(sample=slice(start, end)).values
        preds_chunk = np.asarray(model.predict(X_chunk)).reshape(-1)
        preds_chunks.append(preds_chunk)
        cx_chunks.append(ds[coord_x_var].isel(sample=slice(start, end)).values)
        cy_chunks.append(ds[coord_y_var].isel(sample=slice(start, end)).values)

    preds = np.concatenate(preds_chunks)
    coord_x = np.concatenate(cx_chunks)
    coord_y = np.concatenate(cy_chunks)
    return preds, coord_x, coord_y


"""
Inference utilities for CatBoost models on large xarray datasets.
"""

from typing import Mapping, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from tqdm.auto import tqdm


def _subset_features(ds, feature_list: Sequence[str], feature_var: str = "X_features"):
    """Subset ds[feature_var] to the provided feature_list order."""
    feats_available = list(ds.coords["feature"].values)
    missing = sorted(set(feature_list) - set(feats_available))
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")
    idx = [feats_available.index(f) for f in feature_list]
    return ds[feature_var].isel(feature=idx)


def _stack_if_grid(ds, feature_var: str = "X_features") -> Tuple[xr.Dataset, str, str]:
    """
    If features are arranged as (feature, y, x), stack to (sample, feature)
    and attach coord_x/coord_y derived from x/y coords.
    """
    arr = ds[feature_var]
    if set(arr.dims) == {"feature", "y", "x"}:
        stacked = arr.stack(sample=("y", "x")).transpose("sample", "feature")
        # Extract coordinates before rebuilding to avoid MultiIndex warnings
        x_vals = stacked["x"].values
        y_vals = stacked["y"].values
        feature_vals = stacked["feature"].values
        sample_idx = np.arange(stacked.sizes["sample"])

        # Rebuild DataArray to drop MultiIndex on sample
        da = xr.DataArray(
            stacked.data,
            dims=("sample", "feature"),
            coords={
                "sample": ("sample", sample_idx),
                "feature": ("feature", feature_vals),
                "coord_x": ("sample", x_vals),
                "coord_y": ("sample", y_vals),
            },
            attrs=stacked.attrs,
        )
        ds_out = xr.Dataset({feature_var: da})
        return ds_out, "coord_x", "coord_y"
    return ds, "coord_x", "coord_y"


def predict_dataset_features(
    model,
    ds,
    config: Mapping[str, object],
    chunk_size: int = 500_000,
    feature_var: str = "X_features",
    coord_x_var: str = "coord_x",
    coord_y_var: str = "coord_y",
    stack_grid: bool = True,
    drop_all_nan: bool = True,
    return_sample_indices: bool = False,
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Run model inference on a large stacked dataset, respecting feature order from config['X_features_list'].

    Args:
        model: Trained CatBoost model with predict().
        ds: xarray Dataset containing feature_var with dims (sample, feature) and coord vars.
        config: Config dict that should include 'X_features_list' for ordering.
        chunk_size: Number of samples per batch.
        feature_var: Name of feature DataArray in ds.
        coord_x_var / coord_y_var: Coordinate variable names for output.
        stack_grid: If True, will stack (feature, y, x) → (sample, feature) before inference.
        drop_all_nan: If True, skip samples whose features are all-NaN; outputs keep NaN in those slots.
        return_sample_indices: If True, also return kept sample indices (relative to stacked order).

    Returns:
        preds (np.ndarray), coord_x (np.ndarray), coord_y (np.ndarray) [, sample_idx (np.ndarray)]
    """
    if "X_features_list" not in config:
        raise ValueError("config missing 'X_features_list' for feature ordering.")

    if stack_grid:
        ds, coord_x_var, coord_y_var = _stack_if_grid(ds, feature_var=feature_var)

    feature_list = list(config["X_features_list"])
    X_sel = _subset_features(ds, feature_list, feature_var=feature_var)

    n_samples = int(ds.sizes["sample"])
    # Preallocate outputs to preserve alignment (NaN where skipped)
    preds_full = np.full(n_samples, np.nan, dtype=float)
    cx_full = ds[coord_x_var].values
    cy_full = ds[coord_y_var].values
    kept_indices = []

    for start in tqdm(
        range(0, n_samples, chunk_size),
        desc="Predicting",
        unit="samples",
        total=(n_samples + chunk_size - 1) // chunk_size,
    ):
        end = min(start + chunk_size, n_samples)
        X_chunk = X_sel.isel(sample=slice(start, end)).values
        if drop_all_nan:
            valid_mask = ~np.isnan(X_chunk).all(axis=1)
        else:
            valid_mask = np.ones(X_chunk.shape[0], dtype=bool)

        if not valid_mask.any():
            continue

        preds_chunk = np.asarray(model.predict(X_chunk[valid_mask])).reshape(-1)
        preds_full[start:end][valid_mask] = preds_chunk
        if return_sample_indices:
            kept_indices.append(np.arange(start, end)[valid_mask])

    if np.isnan(preds_full).all():
        raise ValueError("No valid samples to predict (all rows were NaN).")

    if return_sample_indices:
        sample_idx = np.concatenate(kept_indices) if kept_indices else np.array([], dtype=int)
        return preds_full, cx_full, cy_full, sample_idx

    return preds_full, cx_full, cy_full


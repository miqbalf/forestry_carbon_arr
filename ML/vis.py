"""
Visualization helpers for ML workflows.
"""

from typing import Mapping, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


def plot_unstack_ds(
    ds_stacked_formatted,
    unstack_vars: str = "X_features",
    var_name_sample: str = "FCD2_1_MEAN",
    y: bool = False,
    y_var: str = "target",
    cmap_custom: Optional[Mapping[Union[int, str], str]] = None,
    stack_grid: bool = False,
):
    """
    Visualize a stacked xarray dataset by unstacking to spatial grid.

    Args:
        ds_stacked_formatted: xarray.Dataset with coords 'coord_x' and 'coord_y'
        unstack_vars: variable name to visualize when y is False
        var_name_sample: feature name inside unstack_vars (when y is False)
        y: if True, render target classes using y_var
        y_var: variable name for target classes when y is True
        cmap_custom: optional mapping class_id -> color
        stack_grid: if True, expects unstack_vars shaped (feature, y, x); will stack to (sample, feature)
    """
    ds_plot = ds_stacked_formatted
    if stack_grid and set(ds_plot[unstack_vars].dims) == {"feature", "y", "x"}:
        arr = ds_plot[unstack_vars]
        stacked = arr.stack(sample=("y", "x")).transpose("sample", "feature")
        ds_plot = ds_plot.copy()
        ds_plot = ds_plot.assign_coords(
            coord_x=("sample", stacked["x"].values),
            coord_y=("sample", stacked["y"].values),
            feature=("feature", stacked["feature"].values),
            sample=("sample", np.arange(stacked.sizes["sample"])),
        )
        ds_plot = ds_plot.assign({unstack_vars: stacked})

    if not y:
        band_sample = ds_plot[unstack_vars].sel(feature=var_name_sample)
    else:
        band_sample = ds_plot[y_var]

    # Check if band_sample is already in grid format (y, x)
    if set(band_sample.dims) == {"y", "x"}:
        # Already in grid format, no need to unstack
        band_grid = band_sample.assign_attrs(
            crs=str(ds_stacked_formatted.attrs.get("crs", "unknown"))
        )
    else:
        # In sample format, need to assign coordinates and unstack
        band_with_xy = band_sample.assign_coords(
            x=("sample", ds_plot["coord_x"].data),
            y=("sample", ds_plot["coord_y"].data),
        )
        band_grid = (
            band_with_xy.set_index(sample=("x", "y")).unstack("sample").assign_attrs(
                crs=str(ds_stacked_formatted.attrs.get("crs", "unknown"))
            )
        )

    fig, ax = plt.subplots(figsize=(10, 8))

    if y:
        vals = np.unique(band_grid.values[~np.isnan(band_grid.values)])
        vals = vals.astype(int)

        use_custom = isinstance(cmap_custom, Mapping) and len(cmap_custom) > 0
        if use_custom:
            color_list = [cmap_custom.get(int(v)) for v in vals]
            use_custom = all(color_list)
        if use_custom:
            cmap = ListedColormap(color_list)
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(vals), 1)))
            cmap = ListedColormap(colors[: len(vals)])

        boundaries = np.concatenate([vals - 0.5, [vals[-1] + 0.5]]) if len(vals) else [-0.5, 0.5]
        norm = BoundaryNorm(boundaries, cmap.N)
        im = band_grid.plot.imshow(x="x", y="y", ax=ax, cmap=cmap, norm=norm, add_colorbar=False)
        im.set_alpha(0.95)
        cbar = fig.colorbar(im, ax=ax, ticks=vals, boundaries=boundaries)
        cbar.ax.set_title("class id", fontsize=9)
        ax.set_title(f"Target classes (CRS {band_grid.attrs['crs']})")
    else:
        band_grid.plot.imshow(x="x", y="y", ax=ax, cmap="viridis")
        ax.set_title(f"{var_name_sample} (CRS {band_grid.attrs['crs']})")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

    return fig, ax


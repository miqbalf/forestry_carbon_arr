"""
Utilities for preparing data for tsfresh feature extraction with ground truth.

This module provides functions for:
- Loading ground truth training data
- Converting polygons to raster masks
- Merging masks into 4D datasets
- Clipping satellite data to sample bounding boxes
- Merging satellite data with ground truth labels
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
from multiprocessing import Pool
import multiprocessing as mp

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import dask.array as da
import rasterio.features
from affine import Affine

logger = logging.getLogger(__name__)

try:
    import gcsfs
    GCSFS_AVAILABLE = True
except ImportError:
    GCSFS_AVAILABLE = False
    logger.warning("gcsfs not available. GCS operations will be limited.")


def standardize_to_stac_convention(ds: xr.Dataset) -> xr.Dataset:
    """
    Standardize xarray Dataset to STAC convention:
    - Y coordinates descending (maxy to miny)
    - X coordinates ascending (minx to maxx)
    
    This ensures consistent coordinate ordering for clipping operations.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize
    
    Returns
    -------
    xarray.Dataset
        Standardized dataset with STAC-compliant coordinate ordering
    """
    ds = ds.copy()
    
    # Check and fix y coordinates (should be descending for STAC)
    y_values = ds.y.values
    if len(y_values) > 1 and y_values[0] < y_values[-1]:  # Ascending - need to reverse
        logger.info("Reversing y dimension to STAC convention (descending)...")
        ds = ds.reindex(y=ds.y[::-1])
    
    # Check and fix x coordinates (should be ascending)
    x_values = ds.x.values
    if len(x_values) > 1 and x_values[0] > x_values[-1]:  # Descending - need to reverse
        logger.info("Reversing x dimension to standard convention (ascending)...")
        ds = ds.reindex(x=ds.x[::-1])
    
    # Verify final state
    y_final = ds.y.values
    x_final = ds.x.values
    y_ok = len(y_final) <= 1 or y_final[0] > y_final[-1]  # Descending or single value
    x_ok = len(x_final) <= 1 or x_final[0] < x_final[-1]  # Ascending or single value
    
    if y_ok and x_ok:
        logger.info("✓ Dataset standardized to STAC convention")
    else:
        logger.warning(f"⚠️  Coordinate ordering may not be correct: y_descending={y_ok}, x_ascending={x_ok}")
    
    return ds


def load_ground_truth_data(
    gcs_path: Optional[str] = None,
    local_path: Optional[str] = None,
    use_existing: bool = True
) -> gpd.GeoDataFrame:
    """
    Load ground truth training data from parquet file (GCS or local).
    
    Parameters
    ----------
    gcs_path : str, optional
        GCS path to parquet file (e.g., 'gs://bucket/path.parquet')
    local_path : str, optional
        Local path to parquet file
    use_existing : bool
        If True, load existing file. If False, expects data to be provided.
        Default True.
    
    Returns
    -------
    gpd.GeoDataFrame
        Ground truth training data with columns: ['layer', 'geometry', 'date', 'type', 'year']
    """
    if not use_existing:
        raise ValueError("use_existing=False not yet implemented. Provide path to existing file.")
    
    if gcs_path:
        if not GCSFS_AVAILABLE:
            raise ImportError("gcsfs is required for GCS operations. Install with: pip install gcsfs")
        
        fs = gcsfs.GCSFileSystem(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
        logger.info(f"Loading ground truth from GCS: {gcs_path}")
        training_gdf = gpd.read_parquet(gcs_path, filesystem=fs)
    elif local_path:
        logger.info(f"Loading ground truth from local: {local_path}")
        training_gdf = gpd.read_parquet(local_path)
    else:
        raise ValueError("Either gcs_path or local_path must be provided")
    
    logger.info(f"Loaded {len(training_gdf)} training polygons")
    logger.info(f"Columns: {list(training_gdf.columns)}")
    logger.info(f"Unique layers: {training_gdf['layer'].unique() if 'layer' in training_gdf.columns else 'N/A'}")
    
    return training_gdf


def get_raster_mask_with_layer(
    date_layer_tuple: Tuple[pd.Timestamp, str],
    ds: xr.Dataset,
    gdf_1_dissolved: gpd.GeoDataFrame,
    gdf_0_dissolved: gpd.GeoDataFrame
) -> xr.DataArray:
    """
    Create a raster mask from vector polygons for a specific date and layer.
    
    Converts polygons to a raster grid matching the satellite data grid.
    Assigns value 1 for tree areas, 0 for non-tree areas, NaN for no data.
    
    Parameters
    ----------
    date_layer_tuple : tuple (date, layer)
        The specific date and layer (e.g., 'sample_1') to process
    ds : xarray.Dataset
        Satellite dataset (to match grid size and coordinates)
    gdf_1_dissolved : GeoDataFrame
        Pre-processed polygons for tree areas (type=1)
    gdf_0_dissolved : GeoDataFrame
        Pre-processed polygons for non-tree areas (type=0)
    
    Returns
    -------
    xarray.DataArray
        A grid with 1s (trees), 0s (non-trees), and NaN (no label)
    """
    date, layer = date_layer_tuple
    
    # Filter polygons for this specific date and layer
    trees = gdf_1_dissolved[
        (gdf_1_dissolved.index.get_level_values('date') == date) & 
        (gdf_1_dissolved.index.get_level_values('layer') == layer)
    ]
    non_trees = gdf_0_dissolved[
        (gdf_0_dissolved.index.get_level_values('date') == date) & 
        (gdf_0_dissolved.index.get_level_values('layer') == layer)
    ]
    
    # Prepare features for rasterization
    features = [(geom, 1) for geom in trees.geometry] + \
               [(geom, 0) for geom in non_trees.geometry]
    
    # Get grid dimensions from satellite dataset
    x = ds.coords['x'].values
    y = ds.coords['y'].values
    
    # Validate that coordinates are not empty
    if len(x) == 0 or len(y) == 0:
        logger.warning(f"Empty dataset for date={date}, layer={layer}. Returning empty mask.")
        # Return empty mask with proper structure
        mask_raster = np.array([], dtype="float32").reshape(0, 0)
        mask_da = xr.DataArray(
            mask_raster,
            dims=("y", "x"),
            coords={
                "y": np.array([], dtype=float),
                "x": np.array([], dtype=float),
                "date": date,
                "layer": layer
            },
        )
        return mask_da
    
    # Handle case where there are no training labels
    if not features:
        mask_raster = np.full((len(y), len(x)), np.nan, dtype="float32")
    else:
        # Calculate pixel resolution
        res_x = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1.0
        res_y = (y[0] - y[-1]) / (len(y) - 1) if len(y) > 1 else 1.0
        
        # Create affine transformation
        transform = Affine.translation(x[0] - res_x / 2, y[0] + res_y / 2) * \
                   Affine.scale(res_x, -res_y)
        
        # Rasterize: convert vector polygons to raster grid
        mask_raster = rasterio.features.rasterize(
            features,
            out_shape=(len(y), len(x)),
            transform=transform,
            fill=np.nan,
            dtype="float32"
        )
    
    # Wrap in xarray DataArray
    mask_da = xr.DataArray(
        mask_raster,
        dims=("y", "x"),
        coords={
            "y": ds.coords["y"],
            "x": ds.coords["x"],
            "date": date,
            "layer": layer
        },
    )
    
    return mask_da


def parallel_rasterize_with_layer(
    date_layer_combinations: List[Tuple[pd.Timestamp, str]],
    ds: xr.Dataset,
    gdf_1_dissolved: gpd.GeoDataFrame,
    gdf_0_dissolved: gpd.GeoDataFrame,
    n_workers: Optional[int] = None
) -> List[xr.DataArray]:
    """
    Process multiple date-layer combinations in parallel.
    
    Parameters
    ----------
    date_layer_combinations : list of tuples
        All (date, layer) pairs to process
    ds : xarray.Dataset
        Satellite dataset
    gdf_1_dissolved, gdf_0_dissolved : GeoDataFrames
        Pre-processed training polygons
    n_workers : int, optional
        Number of parallel processes. If None, uses min(cpu_count/2, len(combinations))
    
    Returns
    -------
    list of xarray.DataArray
        One mask for each date-layer combination
    """
    if n_workers is None:
        n_workers = int(min(mp.cpu_count(), len(date_layer_combinations)) / 2)
        n_workers = max(1, n_workers)  # At least 1 worker
    
    logger.info(f"Rasterizing {len(date_layer_combinations)} date-layer combinations with {n_workers} workers")
    
    # Create partial function with pre-filled arguments
    func = partial(
        get_raster_mask_with_layer,
        ds=ds,
        gdf_1_dissolved=gdf_1_dissolved,
        gdf_0_dissolved=gdf_0_dissolved
    )
    
    # Process in parallel
    with Pool(n_workers) as pool:
        masks = pool.map(func, date_layer_combinations)
    
    # Filter out empty masks
    valid_masks = [mask for mask in masks if mask.size > 0]
    if len(valid_masks) < len(masks):
        logger.warning(f"Filtered out {len(masks) - len(valid_masks)} empty masks")
    
    logger.info(f"Created {len(valid_masks)} valid masks")
    return valid_masks


def merge_all_masks_4d(
    masks: List[xr.DataArray],
    epsg: int = 32749
) -> xr.Dataset:
    """
    Merge masks into a 4D dataset with dimensions (plot_id, time, y, x).
    
    Parameters
    ----------
    masks : list of xarray.DataArray
        Masks with date and layer coordinates
    epsg : int
        EPSG code for coordinate reference system. Default 32749 (UTM Zone 49S).
    
    Returns
    -------
    xarray.Dataset
        Dataset with 'ground_truth' variable, dimensions (plot_id, time, y, x)
    """
    # Filter out empty masks
    valid_masks = [mask for mask in masks if mask.size > 0]
    if len(valid_masks) == 0:
        raise ValueError("No valid masks to merge. All masks are empty.")
    
    if len(valid_masks) < len(masks):
        logger.warning(f"Filtered out {len(masks) - len(valid_masks)} empty masks before merging")
    
    # Collect all dates and plots
    date_plot_pairs = []
    for mask in valid_masks:
        date = pd.Timestamp(mask.coords['date'].values)
        layer = mask.coords['layer'].values.item()
        date_plot_pairs.append((date, layer, mask))
    
    # Get unique plots and times
    unique_plots = sorted(set([plot for _, plot, _ in date_plot_pairs]))
    unique_times = sorted(set([date for date, _, _ in date_plot_pairs]))
    
    logger.info(f"Organizing into 4D dataset:")
    logger.info(f"  {len(masks)} total date-layer combinations")
    logger.info(f"  {len(unique_plots)} unique plots")
    logger.info(f"  {len(unique_times)} unique times")
    logger.info(f"  Target dimensions: (plot_id, time, y, x)")
    
    # Build dict of plot -> time -> mask
    plot_time_masks = {}
    for date, plot_id, mask_orig in date_plot_pairs:
        if plot_id not in plot_time_masks:
            plot_time_masks[plot_id] = {}
        
        # Clean and make lazy
        mask_clean = mask_orig.drop_vars(['date', 'layer'], errors='ignore')
        if not isinstance(mask_clean.data, da.Array):
            mask_clean = mask_clean.chunk({'y': 512, 'x': 512})
        
        plot_time_masks[plot_id][date] = mask_clean
    
    logger.info("  ✓ Prepared masks as dask arrays (lazy)")
    
    # Get spatial dimensions from first mask
    first_mask = masks[0].drop_vars(['date', 'layer'], errors='ignore')
    y_coords = first_mask.y.values
    x_coords = first_mask.x.values
    
    # Create list of DataArrays for each plot
    logger.info("  Creating 4D structure...")
    plot_arrays = []
    
    for plot_id in unique_plots:
        time_arrays = []
        for time in unique_times:
            if time in plot_time_masks[plot_id]:
                time_arrays.append(plot_time_masks[plot_id][time])
            else:
                # Create NaN-filled dask array for missing time
                nan_shape = (len(y_coords), len(x_coords))
                nan_dask = da.full(nan_shape, np.nan, dtype='float32', chunks=(512, 512))
                nan_array = xr.DataArray(
                    nan_dask,
                    dims=['y', 'x'],
                    coords={'y': y_coords, 'x': x_coords}
                )
                time_arrays.append(nan_array)
        
        # Concatenate along time for this plot (lazy)
        plot_array = xr.concat(time_arrays, dim='time')
        plot_arrays.append(plot_array)
        logger.info(f"    {plot_id}: {len(time_arrays)} times")
    
    logger.info("  ✓ Built plot-time structure")
    
    # Concatenate along plot dimension (still lazy)
    logger.info("  Concatenating plots...")
    combined = xr.concat(plot_arrays, dim='plot_id')
    
    # Assign coordinates
    combined = combined.assign_coords({
        'plot_id': unique_plots,
        'time': unique_times,
        'y': y_coords,
        'x': x_coords,
        'epsg': np.int64(epsg)
    })
    
    # Transpose to desired order: (plot_id, time, y, x)
    combined = combined.transpose('plot_id', 'time', 'y', 'x')
    
    # Create Dataset
    gt = xr.Dataset({
        'ground_truth': combined
    })
    
    # Add metadata
    gt.attrs['description'] = 'Ground truth training masks'
    gt.attrs['values'] = '0=non-tree, 1=tree, NaN=no label'
    gt.attrs['crs'] = f'EPSG:{epsg}'
    gt['ground_truth'].attrs['units'] = 'category'
    gt.coords['epsg'].attrs['description'] = 'EPSG code for coordinate reference system'
    
    # Summary
    total_cells = len(unique_plots) * len(unique_times)
    filled_cells = len(date_plot_pairs)
    missing_cells = total_cells - filled_cells
    
    logger.info(f"  ✓ Final dimensions: {dict(gt.dims)}")
    logger.info(f"  ✓ Unique plots: {unique_plots}")
    logger.info(f"  ✓ Data type: {type(gt['ground_truth'].data)}")
    logger.info(f"  ✓ Filled combinations: {filled_cells}/{total_cells} ({100*filled_cells/total_cells:.1f}%)")
    logger.info(f"  ✓ Missing (NaN): {missing_cells} ({100*missing_cells/total_cells:.1f}%)")
    
    return gt


def prepare_tsfresh_data_with_ground_truth(
    ds_resampled: xr.Dataset,
    training_gdf: gpd.GeoDataFrame,
    buffer_pixels: int = 50,
    chunk_sizes: Optional[Dict[str, int]] = None
) -> List[xr.Dataset]:
    """
    Prepare satellite data with ground truth labels for tsfresh feature extraction.
    
    This function:
    1. Clips satellite data to sample bounding boxes
    2. Converts training polygons to raster masks
    3. Merges masks into 4D dataset
    4. Merges satellite data with ground truth labels
    
    Parameters
    ----------
    ds_resampled : xarray.Dataset
        Satellite dataset with dimensions (time, x, y) or (time, X, Y)
    training_gdf : GeoDataFrame
        Ground truth training data with columns: ['layer', 'geometry', 'date', 'type']
    buffer_pixels : int
        Buffer size in pixels around sample bounding boxes. Default 50.
    chunk_sizes : dict, optional
        Chunk sizes for output dataset. Default: {'plot_id': 1, 'time': 20, 'x': 128, 'y': 128}
    
    Returns
    -------
    list of xarray.Dataset
        One dataset per sample (layer), each with dimensions (plot_id, time, x, y)
        and variables: [EVI, NDVI, ground_truth, gt_valid]
    """
    # Normalize dimension names (handle both X/Y and x/y)
    if 'X' in ds_resampled.dims:
        ds_resampled = ds_resampled.rename({'X': 'x', 'Y': 'y'})
    
    # IMPORTANT: Standardize to STAC convention BEFORE clipping
    # This ensures y is descending and x is ascending for consistent slicing
    logger.info("Standardizing dataset to STAC convention...")
    ds_resampled = standardize_to_stac_convention(ds_resampled)
    
    if chunk_sizes is None:
        chunk_sizes = {'plot_id': 1, 'time': 20, 'x': 128, 'y': 128}
    
    unique_layers = training_gdf['layer'].unique()
    logger.info(f"Processing {len(unique_layers)} samples: {list(unique_layers)}")
    
    ds_gt_list = []
    
    # Process each sample separately
    for layer_name in unique_layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {layer_name}")
        logger.info(f"{'='*60}")
        
        # 1. Get geometries for this sample
        sample_gdf = training_gdf[training_gdf['layer'] == layer_name]
        logger.info(f"  Polygons: {len(sample_gdf)}")
        
        # 2. Get bounding box with buffer
        bounds = sample_gdf.total_bounds
        pixel_size = float(ds_resampled.x[1] - ds_resampled.x[0])
        buffer_distance = buffer_pixels * pixel_size
        
        minx, miny, maxx, maxy = bounds
        minx -= buffer_distance
        miny -= buffer_distance
        maxx += buffer_distance
        maxy += buffer_distance
        
        # Check if bbox intersects with dataset bounds
        ds_minx = float(ds_resampled.x.min())
        ds_maxx = float(ds_resampled.x.max())
        ds_miny = float(ds_resampled.y.min())
        ds_maxy = float(ds_resampled.y.max())
        
        # Check intersection
        bbox_intersects = not (maxx < ds_minx or minx > ds_maxx or maxy < ds_miny or miny > ds_maxy)
        
        if not bbox_intersects:
            logger.warning(f"  ⚠️  Bounding box does not intersect with dataset for {layer_name}. Skipping this sample.")
            logger.warning(f"     Sample bbox: [{minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}]")
            logger.warning(f"     Dataset bounds: x=[{ds_minx:.2f}, {ds_maxx:.2f}], y=[{ds_miny:.2f}, {ds_maxy:.2f}]")
            continue
        
        # 3. Clip ds_resampled to this bbox
        # Ensure slice bounds are within dataset bounds
        clip_minx = max(minx, ds_minx)
        clip_maxx = min(maxx, ds_maxx)
        clip_miny = max(miny, ds_miny)
        clip_maxy = min(maxy, ds_maxy)
        
        # 3. Clip ds_resampled to this bbox
        # After standardization, y is descending (maxy to miny) and x is ascending (minx to maxx)
        # So we slice: x from minx to maxx, y from maxy to miny
        
        # Find nearest coordinate indices for more robust clipping
        x_coords = ds_resampled.x.values
        y_coords = ds_resampled.y.values
        
        # Find indices where coordinates are within bbox
        # For x (ascending): find indices where x >= clip_minx and x <= clip_maxx
        x_mask = (x_coords >= clip_minx) & (x_coords <= clip_maxx)
        # For y (descending): find indices where y <= clip_maxy and y >= clip_miny
        y_mask = (y_coords <= clip_maxy) & (y_coords >= clip_miny)
        
        if not x_mask.any() or not y_mask.any():
            # Try with tolerance (half pixel size) in case of floating point precision issues
            pixel_size = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 10.0
            tolerance = pixel_size / 2.0
            
            x_mask = (x_coords >= clip_minx - tolerance) & (x_coords <= clip_maxx + tolerance)
            y_mask = (y_coords <= clip_maxy + tolerance) & (y_coords >= clip_miny - tolerance)
            
            if not x_mask.any() or not y_mask.any():
                logger.warning(f"  ⚠️  No coordinates found within bbox for {layer_name}. Skipping this sample.")
                logger.warning(f"     Clip bbox: [{clip_minx:.2f}, {clip_miny:.2f}, {clip_maxx:.2f}, {clip_maxy:.2f}]")
                logger.warning(f"     Dataset bounds: x=[{ds_minx:.2f}, {ds_maxx:.2f}], y=[{ds_miny:.2f}, {ds_maxy:.2f}]")
                logger.warning(f"     X coords range: [{x_coords.min():.2f}, {x_coords.max():.2f}], pixel size: {pixel_size:.2f}")
                logger.warning(f"     Y coords range: [{y_coords.min():.2f}, {y_coords.max():.2f}]")
                logger.warning(f"     Y coordinate order: {'descending' if y_coords[0] > y_coords[-1] else 'ascending'}")
                continue
        
        # Get the actual coordinate ranges that exist in the dataset
        x_indices = np.where(x_mask)[0]
        y_indices = np.where(y_mask)[0]
        
        x_start_idx = x_indices[0]
        x_end_idx = x_indices[-1] + 1  # +1 because slice is exclusive at end
        y_start_idx = y_indices[0]
        y_end_idx = y_indices[-1] + 1
        
        # Use isel for more reliable clipping based on indices
        try:
            ds_clipped = ds_resampled.isel(
                x=slice(x_start_idx, x_end_idx),
                y=slice(y_start_idx, y_end_idx)
            )
        except Exception as e:
            # Fallback to sel if isel fails
            logger.warning(f"  ⚠️  Error with isel, trying sel: {e}")
            try:
                ds_clipped = ds_resampled.sel(
                    x=slice(clip_minx, clip_maxx),
                    y=slice(clip_maxy, clip_miny)  # STAC convention: y descending
                )
            except Exception as e2:
                logger.error(f"  ❌ Failed to clip dataset: {e2}")
                continue
        
        # Validate that clipped dataset is not empty
        if ds_clipped.sizes.get('x', 0) == 0 or ds_clipped.sizes.get('y', 0) == 0:
            logger.warning(f"  ⚠️  Clipped dataset is empty for {layer_name}. Skipping this sample.")
            logger.warning(f"     Clip bbox: [{clip_minx:.2f}, {clip_miny:.2f}, {clip_maxx:.2f}, {clip_maxy:.2f}]")
            logger.warning(f"     Dataset bounds: x=[{ds_minx:.2f}, {ds_maxx:.2f}], y=[{ds_miny:.2f}, {ds_maxy:.2f}]")
            continue
        
        original_size = ds_resampled.sizes['x'] * ds_resampled.sizes['y']
        clipped_size = ds_clipped.sizes['x'] * ds_clipped.sizes['y']
        reduction = 100 * (1 - clipped_size / original_size)
        logger.info(f"  Size: {dict(ds_clipped.sizes)} ({reduction:.1f}% reduction)")
        
        # 4. Get date-layer combinations for THIS sample only
        sample_date_layer_combos = list(
            sample_gdf[['date', 'layer']]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        logger.info(f"  Date-layer combinations: {len(sample_date_layer_combos)}")
        
        # 5. Pre-process: Dissolve geometries by date and layer
        gdf_1_dissolved = sample_gdf[sample_gdf['type'] == 1].dissolve(by=['date', 'layer'])
        gdf_0_dissolved = sample_gdf[sample_gdf['type'] == 0].dissolve(by=['date', 'layer'])
        logger.info(f"  Tree groups: {len(gdf_1_dissolved)}, Non-tree groups: {len(gdf_0_dissolved)}")
        
        # 6. Create masks in parallel
        n_workers = int(min(mp.cpu_count(), len(sample_date_layer_combos)) / 2)
        n_workers = max(1, n_workers)
        logger.info(f"  Creating masks ({n_workers} workers)...")
        masks = parallel_rasterize_with_layer(
            sample_date_layer_combos,
            ds_clipped,
            gdf_1_dissolved,
            gdf_0_dissolved,
            n_workers=n_workers
        )
        logger.info(f"  ✓ Created {len(masks)} masks")
        
        # 7. Merge masks into 4D dataset
        logger.info("  Merging masks...")
        gt = merge_all_masks_4d(masks, epsg=32749)
        logger.info(f"  ✓ GT shape: {dict(gt.sizes)}")
        
        # Verify plot_id matches
        if 'plot_id' in gt.dims:
            current_plot_id = gt.coords['plot_id'].values[0]
            if current_plot_id != layer_name:
                logger.warning(f"  ⚠️  Warning: plot_id mismatch ({current_plot_id} != {layer_name})")
                gt = gt.assign_coords(plot_id=[layer_name])
        else:
            gt = gt.expand_dims(plot_id=[layer_name])
        
        # 8. Create validity masks (lazy)
        gt['gt_valid'] = gt['ground_truth'].notnull().all(dim='time')
        gt['gt_valid'].attrs['description'] = 'Pixels with labels for all times (per plot)'
        
        # 9. Merge with clipped satellite data
        logger.info("  Merging satellite data with ground truth...")
        
        # Add plot_id dimension to ds_clipped before merging
        ds_clipped_expanded = ds_clipped.expand_dims(plot_id=[layer_name])
        
        logger.info(f"    ds_clipped dims: {dict(ds_clipped.sizes)}")
        logger.info(f"    ds_clipped_expanded dims: {dict(ds_clipped_expanded.sizes)}")
        logger.info(f"    gt dims: {dict(gt.sizes)}")
        
        # Merge
        ds_with_gt = xr.merge([
            ds_clipped_expanded,  # EVI, NDVI (now with plot_id)
            gt                    # ground_truth, gt_valid (already has plot_id)
        ], compat='override')
        
        logger.info(f"    ✓ Merged successfully!")
        logger.info(f"    Merged dims: {dict(ds_with_gt.sizes)}")
        logger.info(f"    Variables: {list(ds_with_gt.data_vars)}")
        
        # 10. Ensure consistent chunking after merge
        logger.info("  Rechunking for consistent storage...")
        ds_with_gt = ds_with_gt.chunk(chunk_sizes)
        
        logger.info(f"  ✓ Final dataset ready: {dict(ds_with_gt.sizes)}")
        logger.info(f"    plot_id: {ds_with_gt.coords['plot_id'].values}")
        
        ds_gt_list.append(ds_with_gt)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSED {len(ds_gt_list)} SAMPLES")
    logger.info(f"{'='*60}")
    
    return ds_gt_list


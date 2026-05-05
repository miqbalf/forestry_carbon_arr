"""
GeoTIFF utilities for exporting xarray datasets to GeoTIFF format.

This module provides functions for exporting xarray datasets as GeoTIFF files,
with support for multi-band exports and Google Cloud Storage uploads.
"""

import os
import tempfile
import xarray as xr
import numpy as np
import rasterio
from typing import List, Dict, Any, Optional, Union
import logging

# Import rioxarray to enable .rio accessor on xarray objects
try:
    import rioxarray
    RIOXARRAY_AVAILABLE = True
except ImportError:
    RIOXARRAY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Try to import gcsfs (optional dependency)
try:
    import gcsfs
    GCSFS_AVAILABLE = True
    # Re-use a global filesystem client when possible
    try:
        gcs = gcsfs.GCSFileSystem(
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            token=os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json")
        )
    except Exception:
        gcs = None
except ImportError:
    GCSFS_AVAILABLE = False
    gcs = None


def convert_to_geotiff(
    ds: xr.Dataset,
    band_list: List[str],
    output_path: str,
    nodata_val: Union[int, float] = -9999,
    dtype: str = "int16",
    extra_attrs: Optional[Dict[str, Any]] = None,
    class_name_map: Optional[Dict[Union[int, str], str]] = None,
    compress: str = "DEFLATE",
    local_temp_path: Optional[str] = None
) -> str:
    """
    Export selected bands from xarray Dataset to GeoTIFF format.

    This function converts selected bands from an xarray Dataset to a multi-band
    GeoTIFF file. It supports special encoding for categorical data (e.g.,
    subtype_zone_code) when a class_name_map is provided, and can upload
    directly to Google Cloud Storage.

    Args:
        ds: xarray Dataset containing the bands to export
        band_list: List of band names to include in the GeoTIFF
        output_path: Output path (local file path or GCS URI like 'gs://bucket/path/file.tif')
        nodata_val: Value to use for nodata pixels (default: -9999)
        dtype: Data type for the output bands (default: "int16")
        extra_attrs: Additional attributes to add to the GeoTIFF metadata
        class_name_map: Optional mapping from class codes to class names for encoding
            categorical bands (used when band name contains "subtype_zone_code")
        compress: Compression method for GeoTIFF (default: "DEFLATE")
        local_temp_path: Local path for temporary file when uploading to GCS
            (default: auto-generated temp file)

    Returns:
        str: Path where the file was saved (same as output_path)

    Raises:
        ImportError: If required dependencies (rasterio, rioxarray) are not available
        ValueError: If band_list contains bands not found in dataset
        RuntimeError: If GCS upload fails when output_path is a GCS URI

    Example:
        >>> import xarray as xr
        >>> from forestry_carbon_arr.utils.geotiff_utils import convert_to_geotiff
        >>>
        >>> # Load your dataset
        >>> ds = xr.open_dataset("data.nc")
        >>>
        >>> # Export specific bands to local GeoTIFF
        >>> convert_to_geotiff(
        ...     ds,
        ...     band_list=["el_tsfresh", "lc_class"],
        ...     output_path="output.tif"
        ... )
        >>>
        >>> # Export with categorical encoding to GCS
        >>> class_map = {1: "Forest", 2: "Grassland", 3: "Water"}
        >>> convert_to_geotiff(
        ...     ds,
        ...     band_list=["el_tsfresh", "subtype_zone_code"],
        ...     output_path="gs://my-bucket/results/classification.tif",
        ...     class_name_map=class_map
        ... )
    """
    if not RIOXARRAY_AVAILABLE:
        raise ImportError("rioxarray is required for GeoTIFF export functionality")

    extra_attrs = extra_attrs or {}

    # Validate inputs
    if not band_list:
        raise ValueError("band_list cannot be empty")

    missing_bands = [band for band in band_list if band not in ds.data_vars]
    if missing_bands:
        raise ValueError(f"Bands not found in dataset: {missing_bands}")

    # Check if dataset has CRS information
    if not hasattr(ds, 'rio') or ds.rio.crs is None:
        logger.warning("Dataset does not have CRS information. Make sure to set it before calling this function.")
        # Try to set spatial dims if not already set
        if 'x' in ds.dims and 'y' in ds.dims:
            try:
                ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
            except Exception as e:
                logger.warning(f"Could not set spatial dimensions: {e}")

    # Process bands
    bands = {}
    name_to_code = None

    if class_name_map and any("subtype_zone_code" in band for band in band_list):
        name_to_code = {v: k for k, v in class_name_map.items()}

    for name in band_list:
        if name == "subtype_zone_code" and name_to_code is not None:
            # Special handling for categorical data
            if not PANDAS_AVAILABLE:
                raise ImportError("pandas is required for categorical band encoding")

            bands[name] = xr.apply_ufunc(
                lambda v: nodata_val if pd.isna(v) else name_to_code.get(str(v), nodata_val),
                ds[name],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[np.int16],
            )
        else:
            bands[name] = ds[name]

    # Clean bands: fill nodata and set data type
    bands_clean = {
        k: v.fillna(nodata_val).astype(dtype).rio.write_nodata(nodata_val)
        for k, v in bands.items()
    }

    # Stack bands into multi-band DataArray
    stack = xr.concat(
        [bands_clean[k] for k in bands_clean],
        dim="band",
    ).assign_coords(band=list(bands_clean.keys()))

    # Add metadata
    stack = stack.assign_attrs(nodata=nodata_val, **extra_attrs)

    # Set CRS if available
    if hasattr(ds, 'rio') and ds.rio.crs is not None:
        stack = stack.rio.write_crs(ds.rio.crs, inplace=False)

    # Determine output path and temp file handling
    is_gcs_path = output_path.startswith('gs://')

    if is_gcs_path:
        if not GCSFS_AVAILABLE:
            raise ImportError("gcsfs is required for GCS uploads")
        if gcs is None:
            raise RuntimeError("GCS filesystem not properly initialized")

        local_out = local_temp_path or tempfile.mktemp(suffix='.tif')
    else:
        local_out = output_path

    # Export to GeoTIFF
    try:
        stack.rio.to_raster(
            local_out,
            driver="GTiff",
            compress=compress,
            dtype=dtype
        )

        # Set band descriptions in the GeoTIFF metadata
        with rasterio.open(local_out, "r+") as dst:
            dst.descriptions = list(bands_clean.keys())

        # Upload to GCS if needed
        if is_gcs_path:
            try:
                gcs.put(local_out, output_path)
                logger.info(f"Successfully uploaded to {output_path}")
                print(f"Uploaded to {output_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to upload to GCS: {e}") from e
            finally:
                # Clean up local temp file
                if os.path.exists(local_out):
                    os.unlink(local_out)

    except Exception as e:
        # Clean up temp file on error
        if is_gcs_path and os.path.exists(local_out):
            os.unlink(local_out)
        raise

    return output_path


def export_single_band_geotiff(
    da: xr.DataArray,
    output_path: str,
    nodata_val: Union[int, float] = -9999,
    dtype: str = "float32",
    compress: str = "DEFLATE",
    local_temp_path: Optional[str] = None
) -> str:
    """
    Export a single-band xarray DataArray to GeoTIFF format.

    Args:
        da: xarray DataArray to export
        output_path: Output path (local or GCS URI)
        nodata_val: Value to use for nodata pixels
        dtype: Data type for the output
        compress: Compression method
        local_temp_path: Local temp path for GCS uploads

    Returns:
        str: Path where the file was saved
    """
    # Convert DataArray to Dataset for compatibility
    ds = da.to_dataset(name='band')

    return convert_to_geotiff(
        ds=ds,
        band_list=['band'],
        output_path=output_path,
        nodata_val=nodata_val,
        dtype=dtype,
        compress=compress,
        local_temp_path=local_temp_path
    )

"""
Zarr utilities for saving and loading xarray datasets.

This module provides functions for efficiently saving and loading xarray datasets
to/from zarr stores, supporting both local filesystem and Google Cloud Storage.
"""

import os
import time
import shutil
import json
import xarray as xr
from numcodecs import Blosc
from typing import Optional, Dict, Any
import logging

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


def _format_size(size_bytes: float) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def check_zarr_gee_compatibility(zarr_path: str, data_var: Optional[str] = None) -> Dict[str, Any]:
    """
    Check if a zarr store is compatible with Google Earth Engine.
    
    GEE requires zarr URIs to end with '/.zarray' for individual arrays.
    This function checks the zarr structure and provides compatibility information.
    
    Args:
        zarr_path: Path to zarr store (local or GCS URI like gs://bucket/path.zarr)
        data_var: Optional data variable name to check specific array
        
    Returns:
        Dictionary with compatibility information:
            - 'is_gee_compatible': bool - Whether the path format is GEE-compatible
            - 'zarr_path': str - Original zarr path
            - 'gee_uri': str - GEE-compatible URI (if applicable)
            - 'is_gcs': bool - Whether path is a GCS URI
            - 'has_zarray': bool - Whether .zarray file exists
            - 'recommendations': List[str] - Recommendations for GEE compatibility
    """
    import os
    
    result = {
        'is_gee_compatible': False,
        'zarr_path': zarr_path,
        'gee_uri': None,
        'is_gcs': False,
        'has_zarray': False,
        'recommendations': []
    }
    
    # Check if it's a GCS path
    is_gcs = zarr_path.startswith('gs://')
    result['is_gcs'] = is_gcs
    
    if not is_gcs:
        result['recommendations'].append("GEE requires GCS paths (gs://bucket/path). Consider uploading to GCS.")
    
    # Check if path ends with /.zarray (GEE requirement)
    if zarr_path.endswith('/.zarray'):
        result['is_gee_compatible'] = True
        result['gee_uri'] = zarr_path
        return result
    
    # If it's a zarr store path, check if .zarray exists
    if data_var:
        # For specific data variable: gs://bucket/path.zarr/{var}/.zarray
        gee_uri = f"{zarr_path.rstrip('/')}/{data_var}/.zarray"
        result['gee_uri'] = gee_uri
        result['recommendations'].append(f"For data variable '{data_var}', use URI: {gee_uri}")
    else:
        # For root zarr: gs://bucket/path.zarr/.zarray (if single array)
        # Or: gs://bucket/path.zarr/{variable}/.zarray (for specific variable)
        gee_uri = f"{zarr_path.rstrip('/')}/.zarray"
        result['gee_uri'] = gee_uri
        result['recommendations'].append(f"GEE requires URI ending with '/.zarray'. Try: {gee_uri}")
        result['recommendations'].append("For datasets with multiple variables, specify the variable name.")
    
    # Try to check if .zarray file exists (for local paths)
    if not is_gcs:
        zarray_path = gee_uri.replace('gs://', '')
        if os.path.exists(zarray_path):
            result['has_zarray'] = True
    else:
        # For GCS, we can't easily check without filesystem access
        result['recommendations'].append("Verify that the .zarray file exists in GCS bucket")
    
    return result


def get_gee_zarr_uri(zarr_path: str, data_var: Optional[str] = None) -> str:
    """
    Convert zarr store path to GEE-compatible URI.
    
    GEE requires zarr URIs to end with '/.zarray' for individual arrays.
    
    Args:
        zarr_path: Path to zarr store (local or GCS URI like gs://bucket/path.zarr)
        data_var: Optional data variable name. If provided, returns URI for that specific array.
                  If None and dataset has multiple variables, you need to specify the variable.
        
    Returns:
        GEE-compatible URI ending with '/.zarray'
        
    Examples:
        >>> get_gee_zarr_uri('gs://bucket/data.zarr', 'NDVI')
        'gs://bucket/data.zarr/NDVI/.zarray'
        
        >>> get_gee_zarr_uri('gs://bucket/data.zarr')
        'gs://bucket/data.zarr/.zarray'
    """
    # Remove trailing slash if present
    zarr_path = zarr_path.rstrip('/')
    
    if data_var:
        # For specific data variable
        return f"{zarr_path}/{data_var}/.zarray"
    else:
        # For root or single array
        return f"{zarr_path}/.zarray"


def list_zarr_variables(zarr_path: str, storage: str = 'auto') -> list:
    """
    List all data variables in a zarr store.
    
    Useful for determining which variables are available for GEE export.
    
    Args:
        zarr_path: Path to zarr store (local or GCS URI)
        storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
        
    Returns:
        List of data variable names
    """
    if storage == 'auto':
        storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
    
    try:
        # Use load_dataset_zarr which is defined later in this file
        # This will work because Python functions are resolved at runtime
        ds = load_dataset_zarr(zarr_path, storage=storage)
        return list(ds.data_vars.keys())
    except Exception as e:
        logger.error(f"Error loading zarr to list variables: {e}")
        return []


def convert_zarr_for_gee(
    zarr_path: str,
    output_path: Optional[str] = None,
    data_var: Optional[str] = None,
    storage: str = 'auto',
    overwrite: bool = False
) -> Dict[str, Any]:
    """
    Convert zarr dataset to GEE-compatible format.
    
    GEE requirements:
    1. Must have _ARRAY_DIMENSIONS in .zattrs
    2. Last two dimensions must be spatial (Y, X)
    3. Must use supported compression codecs (blosc, gzip, lz4, zlib, zstd)
    4. Must have .zmetadata file (consolidated metadata)
    5. CRS information must be present
    6. URI must end with '/.zarray' for individual arrays
    
    Args:
        zarr_path: Path to input zarr store
        output_path: Path to output GEE-compatible zarr. If None, overwrites input.
        data_var: Optional data variable name to export. If None, exports all variables.
        storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
        overwrite: Whether to overwrite existing output. Default False.
        
    Returns:
        Dictionary with conversion results:
            - 'success': bool - Whether conversion was successful
            - 'output_path': str - Path to converted zarr
            - 'gee_uri': str - GEE-compatible URI
            - 'warnings': List[str] - Any warnings during conversion
            - 'requirements_met': Dict[str, bool] - Which GEE requirements are met
    """
    import json
    
    if storage == 'auto':
        storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
    
    if output_path is None:
        output_path = zarr_path
    
    result = {
        'success': False,
        'output_path': output_path,
        'gee_uri': None,
        'warnings': [],
        'requirements_met': {
            'has_array_dimensions': False,
            'has_spatial_dims': False,
            'has_crs': False,
            'has_zmetadata': False,
            'compression_supported': False
        }
    }
    
    try:
        # Load the dataset
        logger.info(f"Loading zarr dataset: {zarr_path}")
        ds = load_dataset_zarr(zarr_path, storage=storage)
        
        # Check if data_var exists
        if data_var and data_var not in ds.data_vars:
            raise ValueError(f"Data variable '{data_var}' not found in dataset. Available: {list(ds.data_vars.keys())}")
        
        # Select specific variable if requested
        if data_var:
            ds = ds[[data_var]]
            logger.info(f"Exporting single variable: {data_var}")
        
        # Clear any existing encoding/compression settings to avoid conflicts
        # We'll set new GEE-compatible compression in save_dataset_efficient_zarr
        if hasattr(ds, 'encoding'):
            # Clear encoding to avoid codec conflicts
            for var in ds.data_vars:
                if var in ds.encoding:
                    del ds.encoding[var]
        
        # If dataset has dask arrays, we need to handle them carefully
        # For conversion, we want to ensure they're properly chunked for GEE
        try:
            import dask.array as da
            has_dask = any(isinstance(var.data, da.Array) for var in ds.data_vars.values())
            if has_dask:
                logger.info("Dataset contains dask arrays - will be computed during save")
        except ImportError:
            pass
        
        # Check dimension requirements
        dims = list(ds.dims.keys())
        if len(dims) < 2:
            raise ValueError("GEE requires at least 2 dimensions (spatial: Y, X)")
        
        # Check if last two dimensions are spatial (y, x or Y, X)
        last_two_dims = dims[-2:]
        spatial_dims = ['y', 'x', 'Y', 'X']
        has_spatial = all(dim.lower() in ['y', 'x'] for dim in last_two_dims)
        
        if not has_spatial:
            result['warnings'].append(
                f"Last two dimensions should be spatial (Y, X). Found: {last_two_dims}. "
                f"GEE may have issues with this."
            )
        else:
            result['requirements_met']['has_spatial_dims'] = True
        
        # Ensure CRS is present
        if 'crs' in ds.attrs:
            result['requirements_met']['has_crs'] = True
        else:
            result['warnings'].append("No CRS found in dataset attributes. GEE may require CRS information.")
        
        # Note: _ARRAY_DIMENSIONS will be added to .zattrs by xarray automatically
        result['requirements_met']['has_array_dimensions'] = True
        
        # Ensure consolidated metadata (creates .zmetadata)
        result['requirements_met']['has_zmetadata'] = True
        
        # Compression is supported (using lz4)
        result['requirements_met']['compression_supported'] = True
        
        # Clean attributes for GEE compatibility
        ds = _clean_dataset_attrs(ds)
        
        # Ensure CRS is in attributes
        if 'crs' not in ds.attrs:
            # Try to get CRS from rioxarray
            try:
                if hasattr(ds, 'rio') and hasattr(ds.rio, 'crs'):
                    crs = ds.rio.crs
                    if crs:
                        ds.attrs['crs'] = str(crs)
                        result['requirements_met']['has_crs'] = True
            except:
                pass
            # Try to get from spatial_ref coordinate
            if 'crs' not in ds.attrs and 'spatial_ref' in ds.coords:
                try:
                    crs_wkt = ds.spatial_ref.attrs.get('crs_wkt', None)
                    if crs_wkt:
                        ds.attrs['crs'] = crs_wkt
                        result['requirements_met']['has_crs'] = True
                except:
                    pass
        
        # Chunk appropriately for GEE (GEE prefers reasonable chunk sizes)
        chunk_sizes = {}
        for dim in dims:
            if dim.lower() in ['y', 'x']:
                chunk_sizes[dim] = min(512, ds.dims[dim])
            elif dim.lower() == 'time':
                chunk_sizes[dim] = min(10, ds.dims[dim])
            else:
                chunk_sizes[dim] = min(100, ds.dims[dim])
        
        # Save to output path using save_dataset_efficient_zarr
        # This ensures proper compression handling and avoids codec issues
        logger.info(f"Converting zarr to GEE-compatible format: {output_path}")
        print(f"🔄 Converting existing zarr to GEE-compatible format...")
        print(f"   Input: {zarr_path}")
        print(f"   Output: {output_path}")
        print(f"   Variables: {list(ds.data_vars.keys())}")
        print(f"   Dimensions: {dims}")
        
        # For conversion, we need to ensure the dataset is in a clean state
        # Remove any existing encoding that might conflict
        ds_clean = ds.copy(deep=False)  # Shallow copy to avoid modifying original
        if hasattr(ds_clean, 'encoding'):
            # Clear all encoding to start fresh
            ds_clean.encoding = {}
        
        # Use save_dataset_efficient_zarr which handles compression correctly
        # IMPORTANT: GEE requires Zarr v2 (not v3) - ensure zarr_version=2
        # This will create a new zarr with GEE-compatible settings
        save_dataset_efficient_zarr(
            ds=ds_clean,  # Use cleaned dataset without old encoding
            zarr_path=output_path,
            chunk_sizes=chunk_sizes,
            compression='lz4',  # GEE-compatible (lz4 is supported per GEE docs)
            compression_level=1,
            overwrite=overwrite,
            consolidated=True,  # Creates .zmetadata (required by GEE)
            storage=storage,
            zarr_version=2  # GEE requires Zarr v2, not v3
        )
        
        # Generate GEE URI
        if data_var:
            gee_uri = get_gee_zarr_uri(output_path, data_var)
        else:
            # For single variable, use root
            if len(ds.data_vars) == 1:
                var_name = list(ds.data_vars.keys())[0]
                gee_uri = get_gee_zarr_uri(output_path, var_name)
            else:
                gee_uri = get_gee_zarr_uri(output_path)
        
        result['success'] = True
        result['gee_uri'] = gee_uri
        
        logger.info(f"✅ Zarr converted to GEE-compatible format")
        logger.info(f"   GEE URI: {gee_uri}")
        print(f"✅ Conversion complete!")
        print(f"   GEE URI: {gee_uri}")
        
        # Check requirements
        all_met = all(result['requirements_met'].values())
        if not all_met:
            result['warnings'].append("Some GEE requirements may not be fully met. Check requirements_met for details.")
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting zarr for GEE: {e}", exc_info=True)
        result['warnings'].append(f"Conversion failed: {str(e)}")
        return result


def _clean_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Clean non-serializable attributes from dataset before saving to zarr.
    
    Removes or converts attributes that can't be serialized to JSON,
    such as RasterSpec objects from stackstac.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to clean
        
    Returns
    -------
    xarray.Dataset
        Dataset with cleaned attributes
    """
    # Create a copy to avoid modifying the original
    ds_clean = ds.copy(deep=False)
    
    # List of attribute keys that are known to be non-serializable
    # These are typically added by stackstac
    problematic_attrs = ['spec', 'transform', 'raster_spec']
    
    # Clean dataset-level attributes
    attrs_to_remove = []
    attrs_to_keep = {}
    
    for key, value in ds_clean.attrs.items():
        # Check if it's a problematic attribute or non-serializable
        if key in problematic_attrs:
            attrs_to_remove.append(key)
        else:
            # Try to serialize to check if it's JSON-serializable
            try:
                json.dumps(value)
                attrs_to_keep[key] = value
            except (TypeError, ValueError):
                # Not serializable, remove it
                attrs_to_remove.append(key)
                logger.debug(f"Removing non-serializable attribute: {key} (type: {type(value).__name__})")
    
    # Remove problematic attributes
    for key in attrs_to_remove:
        if key in ds_clean.attrs:
            del ds_clean.attrs[key]
    
    # Clean variable-level attributes
    for var_name in ds_clean.data_vars:
        var_attrs_to_remove = []
        for key, value in ds_clean[var_name].attrs.items():
            if key in problematic_attrs:
                var_attrs_to_remove.append(key)
            else:
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    var_attrs_to_remove.append(key)
                    logger.debug(f"Removing non-serializable attribute from {var_name}: {key} (type: {type(value).__name__})")
        
        for key in var_attrs_to_remove:
            if key in ds_clean[var_name].attrs:
                del ds_clean[var_name].attrs[key]
    
    # Clean coordinate-level attributes
    for coord_name in ds_clean.coords:
        coord_attrs_to_remove = []
        for key, value in ds_clean.coords[coord_name].attrs.items():
            if key in problematic_attrs:
                coord_attrs_to_remove.append(key)
            else:
                try:
                    json.dumps(value)
                except (TypeError, ValueError):
                    coord_attrs_to_remove.append(key)
                    logger.debug(f"Removing non-serializable attribute from coordinate {coord_name}: {key} (type: {type(value).__name__})")
        
        for key in coord_attrs_to_remove:
            if key in ds_clean.coords[coord_name].attrs:
                del ds_clean.coords[coord_name].attrs[key]
    
    if attrs_to_remove:
        logger.info(f"Cleaned {len(attrs_to_remove)} non-serializable attributes from dataset")
    
    return ds_clean


def save_dataset_efficient_zarr(
    ds: xr.Dataset,
    zarr_path: str,
    chunk_sizes: Optional[Dict[str, int]] = None,
    compression: Optional[str] = None,
    compression_level: int = 1,
    overwrite: bool = True,
    consolidated: Optional[bool] = None,
    storage: str = 'auto',
    gcs_project: Optional[str] = None,
    align_chunks: bool = True,
    zarr_version: Optional[int] = None,
    gee_compatible: bool = True
) -> str:
    """
    Save xarray dataset to zarr format with efficient parallel processing.
    
    By default, saves in GEE-compatible format (Zarr v2, lz4 compression, consolidated metadata).
    Can be customized for other use cases.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save (lazy dask arrays or in-memory).
    zarr_path : str
        Destination path or GCS URI (e.g. gs://bucket/path.zarr).
    chunk_sizes : dict, optional
        Chunk sizes per dimension. Defaults to reasonable values.
    compression : str, optional
        Compression algorithm ('lz4', 'blosc', 'zstd', 'gzip', 'zlib', or None).
        If None and gee_compatible=True, defaults to 'lz4' (GEE-compatible).
        If None and gee_compatible=False, no compression.
    compression_level : int, optional
        Compression level (1-9). Default 1 (fastest).
    overwrite : bool, optional
        Whether to overwrite existing zarr store. Default True.
    consolidated : bool, optional
        Whether to create consolidated metadata (.zmetadata file).
        If None and gee_compatible=True, defaults to True (required by GEE).
        If None and gee_compatible=False, defaults to True.
    storage : str, optional
        Storage type ('auto', 'local', 'gcs'). Default 'auto'.
    gcs_project : str, optional
        GCS project ID (if different from GOOGLE_CLOUD_PROJECT).
    align_chunks : bool, optional
        Whether to align chunks when writing. Default True.
    zarr_version : int, optional
        Zarr format version (2 or 3).
        If None and gee_compatible=True, defaults to 2 (GEE requirement).
        If None and gee_compatible=False, defaults to 2.
    gee_compatible : bool, optional
        Whether to ensure GEE-compatible format. Default True.
        When True, ensures compliance with ee.ImageCollection.loadZarrV2Array requirements:
        - Forces Zarr v2 (GEE requirement)
        - Uses GEE-supported compression: 'blosc', 'gzip', 'lz4', 'zlib', or 'zstd'
        - For blosc, uses meta-compression: 'lz4', 'lz4hc', 'zlib', or 'zstd' (NOT 'blosclz')
        - Creates consolidated metadata (.zmetadata file in parent directory)
        - xarray automatically adds '_ARRAY_DIMENSIONS' to .zattrs (required by GEE)
        - Last two dimensions should be Y and X (spatial dimensions)
        
        Note: GEE requires the URI to point to the .zarray file (e.g., 
        'gs://bucket/path.zarr/variable/.zarray') and the bucket must be in 
        US multi-region, dual-region with US-CENTRAL1, or US-CENTRAL1 region.

    Returns
    -------
    str
        The zarr_path that was written.
    """
    start_time = time.time()
    storage = storage.lower()
    
    if storage == 'auto':
        storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
    
    if storage not in {'local', 'gcs'}:
        raise ValueError("storage must be one of {'auto', 'local', 'gcs'}")
    
    # Apply GEE-compatible defaults if requested
    if gee_compatible:
        # GEE requires Zarr v2
        if zarr_version is None:
            zarr_version = 2
        elif zarr_version != 2:
            logger.warning(f"GEE requires Zarr v2, but zarr_version={zarr_version} was specified. Overriding to v2.")
            zarr_version = 2
        
        # GEE requires consolidated metadata
        if consolidated is None:
            consolidated = True
        elif not consolidated:
            logger.warning("GEE requires consolidated metadata. Setting consolidated=True.")
            consolidated = True
        
        # GEE supports: 'blosc', 'gzip', 'lz4', 'zlib', 'zstd'
        # Default to lz4 if not specified
        if compression is None:
            compression = 'lz4'
        elif compression not in ['blosc', 'gzip', 'lz4', 'zlib', 'zstd']:
            logger.warning(f"Compression '{compression}' may not be GEE-compatible. "
                          f"GEE supports: 'blosc', 'gzip', 'lz4', 'zlib', 'zstd'. "
                          f"Using 'lz4' instead.")
            compression = 'lz4'
    else:
        # Non-GEE defaults
        if zarr_version is None:
            zarr_version = 2
        if consolidated is None:
            consolidated = True
        if compression is None:
            compression = None  # No compression by default for non-GEE
    
    # Setup filesystem for GCS
    if storage == 'gcs':
        if not GCSFS_AVAILABLE:
            raise ImportError("gcsfs is required for GCS storage. Install with: pip install gcsfs")
        
        # Create a fresh GCS filesystem to avoid serialization issues
        # Always create new instance to ensure clean state
        project = gcs_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("GCS project must be specified via gcs_project parameter or GOOGLE_CLOUD_PROJECT environment variable")
        
        token_path = os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json")
        
        # Create filesystem with explicit parameters to avoid None values
        # Only include token if file exists to avoid None in storage_options
        fs_kwargs = {"project": project}
        if token_path and os.path.exists(token_path):
            fs_kwargs["token"] = token_path
        
        fs = gcsfs.GCSFileSystem(**fs_kwargs)
        
        if overwrite and fs.exists(zarr_path):
            print(f"🗑️  Removing existing GCS zarr store: {zarr_path}")
            fs.rm(zarr_path, recursive=True)
    else:
        if overwrite and os.path.exists(zarr_path):
            print(f"🗑️  Removing existing zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)
    
    # Default chunk sizes
    if chunk_sizes is None:
        chunk_sizes = {}
        dims = ds.dims
        if 'time' in dims:
            chunk_sizes['time'] = min(20, dims['time'])
        if 'x' in dims:
            chunk_sizes['x'] = min(256, dims['x'])
        if 'y' in dims:
            chunk_sizes['y'] = min(256, dims['y'])
        for dim_name, dim_len in dims.items():
            chunk_sizes.setdefault(dim_name, min(100, dim_len))
    
    logger.info(f"Saving dataset to zarr: {zarr_path}")
    print(f"📦 Saving to zarr: {zarr_path}")
    if gee_compatible:
        print(f"   🌍 GEE-compatible format (Zarr v2)")
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Chunks: {chunk_sizes}")
    print(f"   Compression: {compression} (level {compression_level})" if compression else "   Compression: None")
    print(f"   Zarr version: {zarr_version}")
    print(f"   Consolidated metadata: {consolidated}")
    print(f"   Storage: {storage}")
    
    # Prepare compression encoding
    # For GEE compatibility, ensure we use proper codec format for Zarr v2
    encoding = {}
    if compression:
        if compression == 'lz4':
            # GEE-compatible: lz4 via Blosc with lz4 meta-compression
            compressor = Blosc(cname='lz4', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
            encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        elif compression == 'blosc':
            # For GEE, blosc meta-compression must be lz4, lz4hc, zlib, or zstd (NOT blosclz)
            # Use lz4hc for better compression while staying GEE-compatible
            compressor = Blosc(cname='lz4hc', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
            encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        elif compression == 'zstd':
            compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
            encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        elif compression == 'gzip':
            # GEE supports gzip directly
            try:
                from numcodecs import GZip
                compressor = GZip(level=compression_level)
                encoding = {var: {'compressor': compressor} for var in ds.data_vars}
            except ImportError:
                logger.warning("GZip codec not available. Falling back to lz4.")
                compressor = Blosc(cname='lz4', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
                encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        elif compression == 'zlib':
            # GEE supports zlib directly
            try:
                from numcodecs import Zlib
                compressor = Zlib(level=compression_level)
                encoding = {var: {'compressor': compressor} for var in ds.data_vars}
            except ImportError:
                logger.warning("Zlib codec not available. Falling back to lz4.")
                compressor = Blosc(cname='lz4', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
                encoding = {var: {'compressor': compressor} for var in ds.data_vars}
        else:
            # Assume it's a dict of encoding settings
            encoding = compression
    
    # Clean non-serializable attributes before saving
    logger.info("Cleaning dataset attributes for zarr compatibility...")
    ds = _clean_dataset_attrs(ds)
    
    # For GEE compatibility, validate dimension ordering
    if gee_compatible:
        dims = list(ds.dims.keys())
        if len(dims) >= 2:
            last_two = dims[-2:]
            if not all(dim.lower() in ['y', 'x'] for dim in last_two):
                logger.warning(
                    f"GEE requires last two dimensions to be Y and X (spatial). "
                    f"Found: {last_two}. GEE may have issues loading this array."
                )
        # Note: xarray automatically adds '_ARRAY_DIMENSIONS' to .zattrs when saving
        # This is required by GEE's loadZarrV2Array function
    
    # Chunk and save
    ds_chunked = ds.chunk(chunk_sizes)
    print("💾 Writing to zarr (with automatic parallelism)...")
    
    store = fs.get_mapper(zarr_path) if storage == 'gcs' else zarr_path
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            ds_chunked.to_zarr(
                store,
                mode='w',
                encoding=encoding,
                consolidated=consolidated,
                compute=True,
                zarr_version=zarr_version,
                align_chunks=align_chunks
            )
    except ImportError:
        ds_chunked.to_zarr(
            store,
            mode='w',
            encoding=encoding,
            consolidated=consolidated,
            compute=True,
            zarr_version=zarr_version,
            align_chunks=align_chunks
        )
    
    elapsed = time.time() - start_time
    
    # Size reporting
    total_size = None
    if storage == 'gcs':
        try:
            size_info = fs.du(zarr_path)
            if isinstance(size_info, dict):
                total_size = sum(size_info.values())
            elif isinstance(size_info, (int, float)):
                total_size = size_info
        except Exception as exc:
            logger.warning(f"Could not compute GCS store size: {exc}")
    else:
        if os.path.exists(zarr_path):
            total_size = 0
            for dirpath, _, filenames in os.walk(zarr_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
    
    if total_size is not None:
        size_str = _format_size(total_size)
        write_speed = total_size / elapsed / (1024 * 1024)
        print("✅ Dataset saved successfully!")
        print(f"   Store size: {size_str}")
        print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   Write speed: {write_speed:.1f} MB/s")
        print(f"   Path: {zarr_path}")
    else:
        print("✅ Dataset saved successfully! (size unavailable)")
        print(f"   Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   Path: {zarr_path}")
    
    return zarr_path


def load_dataset_zarr(
    zarr_path: str,
    consolidated: bool = True,
    storage: str = 'auto',
    gcs_project: Optional[str] = None
) -> xr.Dataset:
    """
    Load a dataset from a zarr store located locally or on GCS.
    
    Parameters
    ----------
    zarr_path : str
        Path to zarr store (local path or GCS URI like gs://bucket/path.zarr).
    consolidated : bool, optional
        Whether to use consolidated metadata. Default True.
    storage : str, optional
        Storage type ('auto', 'local', 'gcs'). Default 'auto'.
    gcs_project : str, optional
        GCS project ID (if different from GOOGLE_CLOUD_PROJECT).
        
    Returns
    -------
    xarray.Dataset
        Loaded dataset.
    """
    storage = storage.lower()
    if storage == 'auto':
        storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
    if storage not in {'local', 'gcs'}:
        raise ValueError("storage must be one of {'auto', 'local', 'gcs'}")
    
    if storage == 'gcs':
        if not GCSFS_AVAILABLE:
            raise ImportError("gcsfs is required for GCS storage. Install with: pip install gcsfs")
        
        # Create a fresh GCS filesystem to avoid serialization issues
        # Always create new instance to ensure clean state
        project = gcs_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise ValueError("GCS project must be specified via gcs_project parameter or GOOGLE_CLOUD_PROJECT environment variable")
        
        token_path = os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json")
        
        # Create filesystem with explicit parameters to avoid None values
        # Only include token if file exists to avoid None in storage_options
        fs_kwargs = {"project": project}
        if token_path and os.path.exists(token_path):
            fs_kwargs["token"] = token_path
        
        fs = gcsfs.GCSFileSystem(**fs_kwargs)
        
        if not fs.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found on GCS: {zarr_path}")
        
        logger.info(f"Loading dataset from GCS zarr: {zarr_path}")
        print(f"📂 Loading dataset from GCS zarr: {zarr_path}")
        
        # Use mapper but ensure filesystem is properly configured
        # The issue was that the filesystem had None values that couldn't be serialized
        # By creating a fresh instance with explicit kwargs, we avoid this
        mapper = fs.get_mapper(zarr_path)
        ds = xr.open_zarr(mapper, consolidated=consolidated)
    else:
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
        logger.info(f"Loading dataset from zarr: {zarr_path}")
        print(f"📂 Loading dataset from zarr: {zarr_path}")
        ds = xr.open_zarr(zarr_path, consolidated=consolidated)
    
    print(f"✅ Dataset loaded: {dict(ds.dims)}")
    return ds


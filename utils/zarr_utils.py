"""
Zarr utilities for saving and loading xarray datasets.

This module provides functions for efficiently saving and loading xarray datasets
to/from zarr stores, supporting both local filesystem and Google Cloud Storage.
"""

import os
import time
import shutil
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


def save_dataset_efficient_zarr(
    ds: xr.Dataset,
    zarr_path: str,
    chunk_sizes: Optional[Dict[str, int]] = None,
    compression: str = 'lz4',
    compression_level: int = 1,
    overwrite: bool = True,
    consolidated: bool = True,
    storage: str = 'auto',
    gcs_project: Optional[str] = None,
    align_chunks: bool = True,
    zarr_version: int = 2
) -> str:
    """
    Simplified zarr saving – focuses on reliable parallelism.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save (lazy dask arrays or in-memory).
    zarr_path : str
        Destination path or GCS URI (e.g. gs://bucket/path.zarr).
    chunk_sizes : dict, optional
        Chunk sizes per dimension. Defaults to reasonable values.
    compression : str, optional
        Compression algorithm ('lz4', 'blosc', 'zstd', or None). Default 'lz4'.
    compression_level : int, optional
        Compression level (1-9). Default 1 (fastest).
    overwrite : bool, optional
        Whether to overwrite existing zarr store. Default True.
    consolidated : bool, optional
        Whether to create consolidated metadata. Default True.
    storage : str, optional
        Storage type ('auto', 'local', 'gcs'). Default 'auto'.
    gcs_project : str, optional
        GCS project ID (if different from GOOGLE_CLOUD_PROJECT).
    align_chunks : bool, optional
        Whether to align chunks when writing. Default True.
    zarr_version : int, optional
        Zarr format version (2 or 3). Default 2.

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
    print(f"   Dimensions: {dict(ds.dims)}")
    print(f"   Chunks: {chunk_sizes}")
    print(f"   Compression: {compression} (level {compression_level})")
    print(f"   Storage: {storage}")
    
    # Prepare compression
    if compression == 'lz4':
        compressor = Blosc(cname='lz4', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
    elif compression == 'blosc':
        compressor = Blosc(cname='blosclz', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
    elif compression == 'zstd':
        compressor = Blosc(cname='zstd', clevel=compression_level, shuffle=Blosc.SHUFFLE, blocksize=0)
        encoding = {var: {'compressor': compressor} for var in ds.data_vars}
    elif compression is None:
        encoding = {}
    else:
        encoding = compression  # assume dict supplied
    
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


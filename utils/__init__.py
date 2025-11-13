"""
Utility modules for Forestry Carbon ARR library.
"""

from .dependency_manager import DependencyManager
from .path_resolver import PathResolver
from .config_loader import ConfigLoader
from .zarr_utils import save_dataset_efficient_zarr, load_dataset_zarr
from .spectral_indices import (
    SpectralIndicesUtils,
    get_index_info,
    formula,
    formula_to_osi_bands,
    formula_osi,
    list_indices
)
from .gee_processing import (
    calculate_utm_crs,
    get_pixel_scale,
    create_image_collection,
    reproject_to_utm,
    prepare_image_collection_for_processing,
    add_cloudm_stats,
    filter_by_cloud_cover,
    create_monthly_composites,
    create_quarterly_composites,
    rename_composite_bands,
    add_spectral_indices,
    add_fcd_indices,
    process_collection_with_indices_and_smoothing,
    remove_drops_and_spikes_gee,
    interpolate_temporal_gaps,
    savgol_filter
)

__all__ = [
    'DependencyManager', 
    'PathResolver', 
    'ConfigLoader',
    'save_dataset_efficient_zarr',
    'load_dataset_zarr',
    'SpectralIndicesUtils',
    'get_index_info',
    'formula',
    'formula_to_osi_bands',
    'formula_osi',
    'list_indices',
    'calculate_utm_crs',
    'get_pixel_scale',
    'create_image_collection',
    'reproject_to_utm',
    'prepare_image_collection_for_processing',
    'add_cloudm_stats',
    'filter_by_cloud_cover',
    'create_monthly_composites',
    'create_quarterly_composites',
    'rename_composite_bands',
    'add_spectral_indices',
    'add_fcd_indices',
    'process_collection_with_indices_and_smoothing',
    'remove_drops_and_spikes_gee',
    'interpolate_temporal_gaps',
    'savgol_filter'
]

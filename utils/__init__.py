"""
Utility modules for Forestry Carbon ARR library.
"""

from .dependency_manager import DependencyManager
from .path_resolver import PathResolver
from .config_loader import ConfigLoader
from .zarr_utils import (
    save_dataset_efficient_zarr, 
    load_dataset_zarr,
    check_zarr_gee_compatibility,
    get_gee_zarr_uri,
    list_zarr_variables,
    convert_zarr_for_gee
)
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
    savgol_filter,
    fill_temporal_gaps_linear
)
from .tsfresh_utils import (
    standardize_to_stac_convention,
    load_ground_truth_data,
    get_raster_mask_with_layer,
    parallel_rasterize_with_layer,
    merge_all_masks_4d,
    prepare_tsfresh_data_with_ground_truth
)
from .stac_utils import (
    parse_stac_id,
    find_duplicate_stac_scenes,
    create_fresh_stac_client,
    search_and_analyze_duplicate_stac_scenes,
    create_unified_stac_visualization,
    debug_stac_scene_data,
    calculate_mpc_date_range
)
from .geotiff_utils import (
    convert_to_geotiff,
    export_single_band_geotiff
)

__all__ = [
    'DependencyManager',
    'PathResolver',
    'ConfigLoader',
    'save_dataset_efficient_zarr',
    'load_dataset_zarr',
    'check_zarr_gee_compatibility',
    'get_gee_zarr_uri',
    'list_zarr_variables',
    'convert_zarr_for_gee',
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
    'savgol_filter',
    'load_ground_truth_data',
    'get_raster_mask_with_layer',
    'parallel_rasterize_with_layer',
    'merge_all_masks_4d',
    'standardize_to_stac_convention',
    'prepare_tsfresh_data_with_ground_truth',
    'parse_stac_id',
    'find_duplicate_stac_scenes',
    'create_fresh_stac_client',
    'search_and_analyze_duplicate_stac_scenes',
    'create_unified_stac_visualization',
    'debug_stac_scene_data',
    'calculate_mpc_date_range',
    'convert_to_geotiff',
    'export_single_band_geotiff'
]

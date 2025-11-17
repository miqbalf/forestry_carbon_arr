"""
Google Earth Engine Processing Utilities

This module provides utilities for processing GEE ImageCollections including:
- ImageCollection creation and initialization
- UTM CRS calculation
- Cloud masking and filtering
- Monthly/quarterly compositing
- Spectral indices addition
- Temporal smoothing and filtering
- Outlier removal
- Time series extraction and visualization
"""

import ee
import pandas as pd
import numpy as np
import logging
import geopandas as gpd
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Optional import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Plotting functions will not work.")


def calculate_utm_crs(aoi_gpd: gpd.GeoDataFrame) -> Tuple[str, int, str]:
    """
    Calculate UTM CRS from AOI centroid.
    
    Parameters
    ----------
    aoi_gpd : gpd.GeoDataFrame
        AOI GeoDataFrame
    
    Returns
    -------
    tuple
        (utm_crs, utm_epsg, hemisphere)
        - utm_crs: CRS string (e.g., 'EPSG:32749')
        - utm_epsg: EPSG code (e.g., 32749)
        - hemisphere: 'N' or 'S'
    """
    # Get centroid for UTM calculation (must be in WGS84)
    if aoi_gpd.crs.to_string() != 'EPSG:4326':
        aoi_wgs84 = aoi_gpd.to_crs('EPSG:4326')
    else:
        aoi_wgs84 = aoi_gpd
    
    centroid_lon = aoi_wgs84.geometry.centroid.x.mean()
    centroid_lat = aoi_wgs84.geometry.centroid.y.mean()
    
    # Calculate UTM zone
    utm_zone = int(np.floor((centroid_lon + 180) / 6)) + 1
    hemisphere = 'N' if centroid_lat >= 0 else 'S'
    
    # UTM EPSG code format: EPSG:326XX (northern) or EPSG:327XX (southern)
    if hemisphere == 'N':
        utm_epsg = 32600 + utm_zone
    else:
        utm_epsg = 32700 + utm_zone
    
    utm_crs = f'EPSG:{utm_epsg}'
    
    logger.info(f"📍 AOI Centroid: ({centroid_lat:.4f}°, {centroid_lon:.4f}°)")
    logger.info(f"🗺️  UTM Zone: {utm_zone}{hemisphere}")
    logger.info(f"🔢 UTM EPSG Code: {utm_crs}")
    
    return utm_crs, utm_epsg, hemisphere


def get_pixel_scale(satellite_type: str) -> int:
    """
    Get pixel scale in meters based on satellite type.
    
    Parameters
    ----------
    satellite_type : str
        Satellite type ('Sentinel', 'Landsat', etc.)
    
    Returns
    -------
    int
        Pixel scale in meters
    """
    if satellite_type == 'Sentinel':
        return 10
    elif satellite_type == 'Landsat':
        return 30
    else:
        return 10  # default


def create_image_collection(
    config: Dict[str, Any],
    aoi_gpd: gpd.GeoDataFrame,
    aoi_ee: ee.Geometry,
    years_back: int = 10,
    use_existing_asset: bool = False,
    asset_folder: Optional[str] = None,
    import_strategy: str = 'container'
) -> ee.ImageCollection:
    """
    Create or get ImageCollection from GEE.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with:
        - I_satellite: Satellite type
        - date_start_end: [start_date, end_date]
        - cloud_cover_threshold: Cloud cover threshold
        - IsThermal: Whether to include thermal bands
    aoi_gpd : gpd.GeoDataFrame
        AOI GeoDataFrame
    aoi_ee : ee.Geometry
        AOI Earth Engine geometry
    years_back : int
        Number of years to look back from end_date. Default 10.
    use_existing_asset : bool
        If True, use existing GEE asset. Default False.
    asset_folder : str, optional
        GEE asset folder path (required if use_existing_asset=True)
    import_strategy : str
        Import strategy ('container', 'local', etc.). Default 'container'.
    
    Returns
    -------
    ee.ImageCollection
        Raw ImageCollection (not mosaicked)
    """
    if use_existing_asset:
        if asset_folder is None:
            raise ValueError("asset_folder is required when use_existing_asset=True")
        ic = ee.ImageCollection(asset_folder)
        logger.info(f"Using existing asset: {asset_folder}")
        return ic
    
    # Create ImageCollection using OSI
    if import_strategy == 'container':
        try:
            from gee_lib.osi.image_collection.main import ImageCollection
        except ImportError:
            from osi.image_collection.main import ImageCollection
    else:
        from GEE_notebook_Forestry.osi.image_collection.main import ImageCollection
    
    # Calculate date range (years_back from end_date)
    year_end = int(config['date_start_end'][1].split('-')[0])
    years_prior = year_end - years_back
    new_date_start_end = [f'{years_prior}-01-01', config['date_start_end'][1]]
    
    logger.info(f"📅 Date range: {new_date_start_end[0]} to {new_date_start_end[1]}")
    
    gee_config = {
        'AOI': aoi_ee,
        'date_start_end': new_date_start_end,
        'cloud_cover_threshold': config['cloud_cover_threshold'],
        'config': {'IsThermal': config.get('IsThermal', False)}
    }
    
    image_collection = ImageCollection(
        I_satellite=config['I_satellite'],
        region=aoi_gpd,
        **gee_config
    )
    
    # Get the raw ImageCollection (not mosaicked)
    ic = image_collection.image_collection_mask()
    logger.info(f"✅ ImageCollection created: {ic.size().getInfo()} images")
    
    return ic


def reproject_to_utm(
    collection: ee.ImageCollection,
    utm_crs: str,
    pixel_scale: int
) -> ee.ImageCollection:
    """
    Reproject ImageCollection to UTM CRS.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection
    utm_crs : str
        UTM CRS string (e.g., 'EPSG:32749')
    pixel_scale : int
        Pixel scale in meters
    
    Returns
    -------
    ee.ImageCollection
        Reprojected collection
    """
    logger.info(f"🔄 Reprojecting collection to UTM...")
    logger.info(f"   CRS: {utm_crs}")
    logger.info(f"   Scale: {pixel_scale}m")
    
    collection_utm = collection.map(
        lambda image: image.reproject(
            crs=utm_crs,
            scale=pixel_scale
        )
    )
    
    logger.info(f"✅ Reprojected to {utm_crs}")
    
    return collection_utm


def prepare_image_collection_for_processing(
    config: Dict[str, Any],
    aoi_gpd: gpd.GeoDataFrame,
    aoi_ee: ee.Geometry,
    years_back: int = 10,
    use_existing_asset: bool = False,
    asset_folder: Optional[str] = None,
    import_strategy: str = 'container',
    reproject_to_utm_flag: bool = True
) -> Tuple[ee.ImageCollection, str, int, str]:
    """
    High-level function to create ImageCollection and prepare it for processing.
    
    This function combines:
    1. Create/get ImageCollection
    2. Calculate UTM CRS
    3. Reproject to UTM (optional)
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    aoi_gpd : gpd.GeoDataFrame
        AOI GeoDataFrame
    aoi_ee : ee.Geometry
        AOI Earth Engine geometry
    years_back : int
        Number of years to look back. Default 10.
    use_existing_asset : bool
        If True, use existing GEE asset. Default False.
    asset_folder : str, optional
        GEE asset folder path (required if use_existing_asset=True)
    import_strategy : str
        Import strategy. Default 'container'.
    reproject_to_utm_flag : bool
        Whether to reproject to UTM. Default True.
    
    Returns
    -------
    tuple
        (collection, utm_crs, pixel_scale, utm_epsg_str)
        - collection: Prepared ImageCollection
        - utm_crs: UTM CRS string (e.g., 'EPSG:32749')
        - pixel_scale: Pixel scale in meters
        - utm_epsg_str: UTM EPSG code as string
    """
    logger.info("=" * 60)
    logger.info("Preparing ImageCollection for processing")
    logger.info("=" * 60)
    
    # Step 1: Create/get ImageCollection
    collection = create_image_collection(
        config=config,
        aoi_gpd=aoi_gpd,
        aoi_ee=aoi_ee,
        years_back=years_back,
        use_existing_asset=use_existing_asset,
        asset_folder=asset_folder,
        import_strategy=import_strategy
    )
    
    # Step 2: Calculate UTM CRS
    utm_crs, utm_epsg, hemisphere = calculate_utm_crs(aoi_gpd)
    pixel_scale = get_pixel_scale(config['I_satellite'])
    
    # Step 3: Reproject to UTM (if requested)
    if reproject_to_utm_flag:
        collection = reproject_to_utm(
            collection,
            utm_crs=utm_crs,
            pixel_scale=pixel_scale
        )
    
    logger.info("✅ ImageCollection prepared for processing")
    
    return collection, utm_crs, pixel_scale, str(utm_epsg)


def add_cloudm_stats(
    image: ee.Image,
    aoi_bounds: ee.Geometry,
    scale: float,
    crs: str
) -> ee.Image:
    """
    Calculate cloudM percentages - only cloudM=1 is valid.
    
    Uses total intersected pixels (valid + invalid) as denominator
    instead of total AOI grid pixels. This accounts for GEE mosaic 
    processing where same tile-ID can appear in multiple scenes.
    
    Uses frequencyHistogram instead of two separate reduceRegion calls
    to reduce concurrent aggregations by 50%.
    
    Parameters
    ----------
    image : ee.Image
        Image with cloudM band
    aoi_bounds : ee.Geometry
        Bounding box geometry in UTM (matches xarray grid extent)
    scale : float
        Pixel scale in meters (same as xarray)
    crs : str
        CRS string (same as xarray)
    
    Returns
    -------
    ee.Image
        Image with added properties:
        - cloudM_pct_valid: Percentage of valid pixels
        - cloudM_count_valid: Count of valid pixels (cloudM=1)
        - cloudM_count_invalid: Count of invalid pixels (cloudM=0)
        - cloudM_total_intersected: Total intersected pixels
    """
    cloudm_band = image.select('cloudM')
    
    # Use frequencyHistogram to get both 0 and 1 counts in ONE reduceRegion call
    histogram = cloudm_band.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=aoi_bounds,
        scale=scale,
        crs=crs,
        bestEffort=True,
        maxPixels=1e11
    ).get('cloudM')
    
    # Extract counts from histogram dictionary
    hist_dict = ee.Dictionary(histogram)
    count_0_raw = hist_dict.get('0', 0)  # Invalid/cloudy pixels
    count_1_raw = hist_dict.get('1', 0)  # Valid pixels
    
    # Convert to ee.Number explicitly
    count_0 = ee.Number(count_0_raw)
    count_1 = ee.Number(count_1_raw)
    
    # Calculate percentage using total intersected pixels
    sum_pixels = count_0.add(count_1)
    pct_valid = count_1.divide(sum_pixels).multiply(100)
    
    return image.set({
        'cloudM_pct_valid': pct_valid,
        'cloudM_count_valid': count_1,
        'cloudM_count_invalid': count_0,
        'cloudM_total_intersected': sum_pixels
    })


def filter_by_cloud_cover(
    collection: ee.ImageCollection,
    aoi_bounds: ee.Geometry,
    scale: float,
    crs: str,
    valid_pixel_threshold: float = 70.0
) -> Tuple[ee.ImageCollection, Dict[str, Any]]:
    """
    Filter ImageCollection by cloud cover percentage.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection with cloudM band
    aoi_bounds : ee.Geometry
        Bounding box geometry in UTM
    scale : float
        Pixel scale in meters
    crs : str
        CRS string
    valid_pixel_threshold : float
        Minimum percentage of valid pixels required (default: 70%)
    
    Returns
    -------
    tuple
        (filtered_collection, stats_dict)
        - filtered_collection: Filtered ImageCollection
        - stats_dict: Dictionary with statistics
    """
    logger.info("Calculating cloudM statistics for each scene...")
    
    # Add statistics to each image
    collection_with_stats = collection.map(
        lambda img: add_cloudm_stats(img, aoi_bounds, scale, crs)
    )
    
    # Get statistics
    logger.info("Computing statistics...")
    pct_array = collection_with_stats.aggregate_array('cloudM_pct_valid').getInfo()
    pct_array = [float(x) if x is not None else 0.0 for x in pct_array]
    
    # Calculate stats
    stats = {}
    if len(pct_array) > 0:
        stats['min'] = min(pct_array)
        stats['max'] = max(pct_array)
        stats['mean'] = sum(pct_array) / len(pct_array)
        stats['n_valid'] = sum(1 for p in pct_array if p > valid_pixel_threshold)
        stats['n_total'] = len(pct_array)
        stats['retention_rate'] = (stats['n_valid'] / stats['n_total'] * 100) if stats['n_total'] > 0 else 0
        
        logger.info(f"Valid pixel percentages: min={stats['min']:.1f}%, max={stats['max']:.1f}%")
        logger.info(f"Mean: {stats['mean']:.1f}%")
        logger.info(f"Scenes with >{valid_pixel_threshold}% valid pixels: {stats['n_valid']}/{stats['n_total']}")
        
        if stats['n_valid'] == 0:
            logger.warning(f"No scenes passed the threshold! Consider lowering valid_pixel_threshold (current: {valid_pixel_threshold}%)")
    else:
        logger.warning("No statistics computed!")
        stats = {'n_valid': 0, 'n_total': 0}
    
    # Filter collection
    collection_filtered = collection_with_stats.filter(
        ee.Filter.gt('cloudM_pct_valid', valid_pixel_threshold)
    )
    
    original_size = collection.size().getInfo()
    filtered_size = collection_filtered.size().getInfo()
    
    logger.info(f"After valid pixel filtering:")
    logger.info(f"  Filtered: {filtered_size} scenes")
    logger.info(f"  Removed: {original_size - filtered_size} scenes")
    logger.info(f"  Retention rate: {stats.get('retention_rate', 0):.1f}%")
    
    return collection_filtered, stats


def create_monthly_composites(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    reducer: str = 'median'
) -> List[ee.Image]:
    """
    Create monthly composites from ImageCollection.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection
    aoi : ee.Geometry
        AOI geometry for clipping
    reducer : str
        Reducer method ('median', 'mean', 'max', 'min')
    
    Returns
    -------
    list
        List of monthly composite ee.Image objects
    """
    logger.info("Creating monthly composites...")
    
    # Get date list
    dates_list = collection.aggregate_array('system:time_start').getInfo()
    logger.info(f"   Input: collection ({len(dates_list)} scenes)")
    logger.info(f"   Output: Monthly composites (each month stamped at the 15th)")
    
    # Create DataFrame for date processing
    dates_df = pd.DataFrame({
        'timestamp': dates_list,
        'date': pd.to_datetime(dates_list, unit='ms')
    })
    dates_df['year'] = dates_df['date'].dt.year
    dates_df['month'] = dates_df['date'].dt.month
    
    # Get unique year-month combinations
    unique_year_month = sorted(
        {(row.year, row.month) for row in dates_df.itertuples()}
    )
    
    # Select reducer
    reducer_func = {
        'median': ee.Reducer.median(),
        'mean': ee.Reducer.mean(),
        'max': ee.Reducer.max(),
        'min': ee.Reducer.min()
    }.get(reducer, ee.Reducer.median())
    
    monthly_images = []
    
    for year, month in unique_year_month:
        start = ee.Date.fromYMD(year, month, 1)
        end = start.advance(1, 'month').advance(-1, 'day')
        
        month_collection = collection.filterDate(start, end.advance(1, 'day'))
        count = month_collection.size().getInfo()
        
        if count == 0:
            continue
        
        # Timestamp at mid-month (15th)
        mid = ee.Date.fromYMD(year, month, 15)
        
        composite = month_collection.reduce(reducer_func).set({
            'system:time_start': mid.millis(),
            'year': year,
            'month': month,
            'n_images': count,
            'system:id': f'Sentinel2_{year}_{month:02d}'
        }).clip(aoi)
        
        monthly_images.append(composite)
        logger.info(f"   {year}-{month:02d}: {count} images → composite "
                   f"(ID: Sentinel2_{year}_{month:02d}, Date stamp: {mid.format('YYYY-MM-dd').getInfo()})")
    
    logger.info(f"✅ Step 1 complete; built {len(monthly_images)} monthly images")
    
    return monthly_images


def rename_composite_bands(
    collection: ee.ImageCollection,
    remove_suffix: str = '_median',
    exclude_bands: Optional[List[str]] = None
) -> ee.ImageCollection:
    """
    Rename bands in composite ImageCollection by removing suffix (e.g., '_median').
    
    After creating monthly composites with a reducer, band names get suffixes
    like 'red_median', 'nir_median', etc. This function removes those suffixes
    and optionally excludes certain bands (e.g., 'cloudM_median').
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection with suffixed band names
    remove_suffix : str
        Suffix to remove from band names. Default '_median'.
    exclude_bands : list, optional
        List of band name patterns to exclude (e.g., ['cloudM']). Default None.
    
    Returns
    -------
    ee.ImageCollection
        Collection with renamed bands (suffix removed)
    """
    logger.info(f"Renaming bands: removing '{remove_suffix}' suffix...")
    
    # Get band names from first image
    first_img = collection.first()
    all_band_names = first_img.bandNames().getInfo()
    
    # Filter out excluded bands
    if exclude_bands:
        list_band_names = [
            band for band in all_band_names 
            if not any(exclude in band for exclude in exclude_bands)
        ]
    else:
        list_band_names = all_band_names
    
    # Create clean band names (remove suffix)
    clean_band_names = [
        band.replace(remove_suffix, '') for band in list_band_names
    ]
    
    logger.info(f"   Original bands: {len(all_band_names)}")
    logger.info(f"   After filtering: {len(list_band_names)}")
    if list_band_names:
        logger.info(f"   Example: {list_band_names[0]} -> {clean_band_names[0]}")
    
    # Rename bands using select with new names
    collection_renamed = collection.map(
        lambda img: img.select(list_band_names, clean_band_names)
    )
    
    logger.info(f"✅ Bands renamed successfully")
    
    return collection_renamed


def create_quarterly_composites(
    collection: ee.ImageCollection,
    aoi: ee.Geometry,
    reducer: str = 'median'
) -> List[ee.Image]:
    """
    Create quarterly composites from ImageCollection.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection
    aoi : ee.Geometry
        AOI geometry for clipping
    reducer : str
        Reducer method ('median', 'mean', 'max', 'min')
    
    Returns
    -------
    list
        List of quarterly composite ee.Image objects
    """
    logger.info("Creating quarterly composites...")
    
    # Get date list
    dates_list = collection.aggregate_array('system:time_start').getInfo()
    logger.info(f"   Input: collection ({len(dates_list)} scenes)")
    logger.info(f"   Output: Quarterly composites (every 3 months, end-of-quarter dates)")
    
    # Get unique years
    years = sorted({pd.Timestamp(ts, unit='ms').year for ts in dates_list})
    
    # Select reducer
    reducer_func = {
        'median': ee.Reducer.median(),
        'mean': ee.Reducer.mean(),
        'max': ee.Reducer.max(),
        'min': ee.Reducer.min()
    }.get(reducer, ee.Reducer.median())
    
    quarterly_images = []
    
    for year in years:
        for quarter in [1, 2, 3, 4]:
            # Quarter start/end
            start_month = (quarter - 1) * 3 + 1
            start = ee.Date.fromYMD(year, start_month, 1)
            end = start.advance(3, 'month').advance(-1, 'day')
            
            quarter_collection = collection.filterDate(start, end.advance(1, 'day'))
            count = quarter_collection.size().getInfo()
            
            if count == 0:
                continue
            
            composite = quarter_collection.reduce(reducer_func).set({
                'system:time_start': end.millis(),
                'year': year,
                'quarter': quarter,
                'n_images': count,
                'system:id': f'Sentinel2_{year}_q{quarter}'
            }).clip(aoi)
            
            quarterly_images.append(composite)
            logger.info(f"   Q{quarter} {year}: {count} images → composite "
                       f"(ID: Sentinel2_{year}_q{quarter}, Date: {end.format('YYYY-MM-dd').getInfo()})")
    
    logger.info(f"✅ Step 1 complete; built {len(quarterly_images)} quarterly images")
    
    return quarterly_images


def add_spectral_indices(
    image: ee.Image,
    config: Dict[str, Any],
    spectral_config: Dict[str, Any]
) -> ee.Image:
    """
    Add spectral indices to a single image using SpectralAnalysis.
    
    Available indices based on satellite type:
    - All: NDVI, NDWI, MSAVI2, MTVI2, VARI
    - Sentinel/Landsat: BSI (requires swir1)
    
    Parameters
    ----------
    image : ee.Image
        Input image
    config : dict
        Main configuration dictionary
    spectral_config : dict
        SpectralAnalysis configuration
    
    Returns
    -------
    ee.Image
        Image with spectral indices added
    """
    try:
        # Try to import SpectralAnalysis
        try:
            from gee_lib.osi.spectral_indices.spectral_analysis import SpectralAnalysis
        except ImportError:
            try:
                from osi.spectral_indices.spectral_analysis import SpectralAnalysis
            except ImportError:
                try:
                    from GEE_notebook_Forestry.osi.spectral_indices.spectral_analysis import SpectralAnalysis
                except ImportError:
                    logger.warning("SpectralAnalysis not available. Skipping spectral indices.")
                    return image
        
        # Create SpectralAnalysis instance
        spectral = SpectralAnalysis(image, spectral_config)
        
        # Start with original image
        image_with_indices = image
        
        # Add simple indices
        ndvi = spectral.NDVI_func()
        ndwi = spectral.NDWI_func()
        msavi2 = spectral.MSAVI2_func()
        mtvi2 = spectral.MTVI2_func()
        vari = spectral.VARI_func()
        
        image_with_indices = image_with_indices.addBands([ndvi, ndwi, msavi2, mtvi2, vari])
        
        # Add BSI for Sentinel/Landsat (requires swir1)
        if config.get('I_satellite') in ['Sentinel', 'Landsat']:
            try:
                bsi = spectral.BSI_func()
                image_with_indices = image_with_indices.addBands(bsi)
            except Exception:
                # BSI requires swir1 - skip if not available
                pass
        
        return image_with_indices
        
    except Exception as e:
        logger.warning(f"Could not add spectral indices: {e}")
        return image


def add_fcd_indices(image: ee.Image) -> ee.Image:
    """
    Add Pseudo Forest Canopy Density (Pseudo-FCD) indices to a single image.
    
    Based on: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/pseudo_forest_canopy_density/
    Uses simplified FCD indices without PCA for computational efficiency.
    
    Parameters
    ----------
    image : ee.Image
        Input image (should already have NDVI and NDWI from spectral indices step)
    
    Returns
    -------
    ee.Image
        Image with FCD indices added (BI, SI, pseudo_fcd_pct)
    """
    try:
        image_with_fcd = image
        
        # Use existing NDVI and NDWI from spectral indices
        ndvi = image.select('NDVI')
        ndwi = image.select('ndwi')
        
        # BI (Bare soil index)
        bi = image.expression(
            '(NIR + Green + Red) / (NIR + Green - Red)', {
                'NIR': image.select('nir'),
                'Green': image.select('green'),
                'Red': image.select('red')
            }
        ).rename('BI')
        
        # SI (Shadow index)
        si = image.expression(
            'pow((1 - Green) * (1 - Red), 0.5)', {
                'Green': image.select('green'),
                'Red': image.select('red')
            }
        ).rename('SI')
        
        # Add BI and SI
        image_with_fcd = image_with_fcd.addBands([bi, si])
        
        # Calculate continuous Pseudo-FCD percentage (0-100%)
        ndvi_s = ndvi.unitScale(0.20, 0.80).clamp(0, 1)
        si_s = si.unitScale(0.90, 0.98).clamp(0, 1)
        bi_inv = ee.Image(1).subtract(bi.unitScale(1.0, 3.0)).clamp(0, 1)
        
        # Weighted combination
        pseudo_fcd_0_1 = (
            ndvi_s.multiply(0.5)
            .add(si_s.multiply(0.3))
            .add(bi_inv.multiply(0.2))
        ).clamp(0, 1)
        
        # Suppress water influence
        water_mask = ndwi.lte(0.2)
        pseudo_fcd_0_1 = pseudo_fcd_0_1.updateMask(water_mask)
        
        # Scale to 0-100% and add as band
        pseudo_fcd_pct = pseudo_fcd_0_1.multiply(100).rename('pseudo_fcd_pct')
        image_with_fcd = image_with_fcd.addBands(pseudo_fcd_pct)
        
        return image_with_fcd
        
    except Exception as e:
        logger.warning(f"Could not add Pseudo-FCD indices: {e}")
        return image


def remove_drops_and_spikes_gee(
    image_collection: ee.ImageCollection,
    band_names: List[str],
    window_size: int = 14,
    threshold_percent: float = 0.1
) -> ee.ImageCollection:
    """
    Remove outliers (drops and spikes) from time series using rolling median.
    
    Parameters
    ----------
    image_collection : ee.ImageCollection
        Input collection
    band_names : list
        List of band names to process
    window_size : int
        Number of images for rolling median window (default: 14)
    threshold_percent : float
        Threshold percentage for outlier detection (default: 0.1 = 10%)
    
    Returns
    -------
    ee.ImageCollection
        Collection with outliers removed
    """
    logger.info(f"Removing outliers using rolling median...")
    logger.info(f"   Outlier window: {window_size} images")
    logger.info(f"   Outlier threshold: {threshold_percent * 100}%")
    logger.info(f"   Bands to process: {band_names}")
    
    collection_list = image_collection.toList(image_collection.size())
    n_images = image_collection.size().getInfo()
    
    def process_image(i):
        center_idx = ee.Number(i).int()
        center_img = ee.Image(collection_list.get(center_idx))
        result_img = center_img
        
        for band_name in band_names:
            center_band = center_img.select([band_name])
            
            # Get window indices
            start_idx = center_idx.subtract(window_size // 2).max(0)
            end_idx = center_idx.add(window_size // 2 + 1).min(n_images)
            
            # Extract window images
            window_indices = ee.List.sequence(start_idx, end_idx.subtract(1))
            window_images = window_indices.map(
                lambda idx: ee.Image(collection_list.get(ee.Number(idx)))
            )
            window_collection = ee.ImageCollection.fromImages(window_images)
            
            # Calculate rolling median
            median_band = window_collection.select([band_name]).median()
            
            # Calculate deviation from median
            deviation = center_band.subtract(median_band).abs()
            median_abs = median_band.abs()
            
            # Threshold: deviation > threshold_percent * median_abs
            threshold = median_abs.multiply(threshold_percent)
            is_outlier = deviation.gt(threshold)
            
            # Replace outliers with median
            cleaned_band = center_band.where(is_outlier, median_band)
            
            result_img = result_img.addBands(cleaned_band.rename([band_name]), None, True)
        
        return result_img
    
    processed = ee.List.sequence(0, n_images - 1).map(process_image)
    result_collection = ee.ImageCollection.fromImages(processed)
    
    logger.info("✅ Outlier removal complete!")
    
    return result_collection


def interpolate_temporal_gaps(
    image_collection: ee.ImageCollection,
    band_names: List[str]
) -> ee.ImageCollection:
    """
    Interpolate missing values in time series using neighboring images.
    Similar to XArray's np.interp() - fills gaps between valid observations.
    
    Parameters
    ----------
    image_collection : ee.ImageCollection
        Input collection
    band_names : list
        List of band names to interpolate
    
    Returns
    -------
    ee.ImageCollection
        Collection with gaps interpolated
    """
    logger.info("Interpolating gaps in time series...")
    
    collection_list = image_collection.toList(image_collection.size())
    n_images = image_collection.size().getInfo()
    
    def process_image(i):
        center_idx = ee.Number(i).int()
        center_img = ee.Image(collection_list.get(center_idx))
        result_img = center_img
        
        for band_name in band_names:
            center_band = center_img.select([band_name])
            
            # Check if current pixel is masked/empty
            is_masked = center_band.mask().Not()
            
            # Get previous and next valid images
            prev_idx = center_idx.subtract(1).max(0)
            prev_img = ee.Image(collection_list.get(prev_idx))
            prev_band = prev_img.select([band_name])
            
            next_idx = center_idx.add(1).min(n_images - 1)
            next_img = ee.Image(collection_list.get(next_idx))
            next_band = next_img.select([band_name])
            
            # Simple interpolation: average of previous and next
            interpolated = prev_band.add(next_band).divide(2)
            
            # Fill masked pixels with interpolated values
            filled_band = center_band.unmask(interpolated)
            
            result_img = result_img.addBands(filled_band.rename([band_name]), None, True)
        
        return result_img
    
    processed = ee.List.sequence(0, n_images - 1).map(process_image)
    result_collection = ee.ImageCollection.fromImages(processed)
    
    logger.info("✅ Gap interpolation complete!")
    
    return result_collection


def process_collection_with_indices_and_smoothing(
    collection: ee.ImageCollection,
    config: Dict[str, Any],
    aoi_ee: ee.Geometry,
    spectral_bands: Optional[List[str]] = None,
    smoothing_window: int = 3,
    smoothing_polyorder: int = 2,
    add_fcd: bool = False,
    outlier_window: int = 14,
    outlier_threshold: float = 0.1
) -> ee.ImageCollection:
    """
    High-level function to add spectral indices and apply smoothing.
    
    This function follows the same workflow as notebook 02b:
    1. Add spectral indices (NDVI, NDWI, MSAVI2, MTVI2, VARI, BSI, EVI)
    2. Optionally add FCD indices
    3. Remove outliers (drops and spikes) using rolling median
    4. Interpolate temporal gaps
    5. Apply Savitzky-Golay smoothing
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Input collection (typically monthly composites)
    config : dict
        Configuration dictionary
    aoi_ee : ee.Geometry
        AOI Earth Engine geometry
    spectral_bands : list, optional
        List of spectral bands to smooth. If None, uses ['NDVI', 'EVI'].
        Note: Only bands that exist in the collection will be smoothed.
    smoothing_window : int
        Window length for Savitzky-Golay smoothing. Default 3.
    smoothing_polyorder : int
        Polynomial order for Savitzky-Golay smoothing. Default 2.
    add_fcd : bool
        Whether to add FCD indices. Default False.
    outlier_window : int
        Window size for outlier detection (rolling median). Default 14.
    outlier_threshold : float
        Threshold percentage for outlier detection (0.1 = 10%). Default 0.1.
    
    Returns
    -------
    ee.ImageCollection
        Collection with spectral indices and smoothing applied
    """
    logger.info("=" * 60)
    logger.info("Processing collection: indices + smoothing")
    logger.info("=" * 60)
    
    # Check if collection has system:time_start property (required for monthly composites)
    try:
        first_img = collection.first()
        time_start = first_img.get('system:time_start').getInfo()
        if time_start is None:
            logger.warning(
                "⚠️  Collection images do not have 'system:time_start' property. "
                "This may cause issues with time-based operations. "
                "Monthly composites created with create_monthly_composites() should have this property set."
            )
        else:
            logger.info(f"✅ Collection has system:time_start property (first image: {pd.to_datetime(time_start, unit='ms').strftime('%Y-%m-%d')})")
    except Exception as e:
        if "does not have a 'system:time_start' property" in str(e) or "system:time_start" in str(e):
            logger.warning(
                "⚠️  Collection images do not have 'system:time_start' property. "
                "This may cause issues with time-based operations. "
                "Monthly composites created with create_monthly_composites() should have this property set. "
                f"Error: {e}"
            )
        else:
            logger.warning(f"⚠️  Could not verify system:time_start property: {e}")
    
    # Prepare spectral config
    spectral_config = {
        'I_satellite': config['I_satellite'],
        'AOI': aoi_ee.geometry(),
        'pca_scaling': config.get('pca_scaling', 1),
        'tileScale': config.get('tileScale', 2)
    }
    
    # Step 1: Add spectral indices
    logger.info("Step 1: Adding spectral indices...")
    collection_with_indices = collection.map(
        lambda img: add_spectral_indices(img, config, spectral_config)
    )
    logger.info("✅ Spectral indices added")
    
    # Step 2: Optionally add FCD indices
    if add_fcd:
        logger.info("Step 2: Adding FCD indices...")
        collection_with_indices = collection_with_indices.map(add_fcd_indices)
        logger.info("✅ FCD indices added")
    
    # Step 3: Set default spectral bands if not provided
    if spectral_bands is None:
        # Default to NDVI and EVI (EVI will be added via eemont)
        spectral_bands = ['NDVI', 'EVI']
    
    # Step 2.5: Add EVI using eemont (if requested in spectral_bands)
    # EVI is not added by SpectralAnalysis, so we use eemont's spectralIndices()
    try:
        import eemont
        # Check if EVI is requested
        if 'EVI' in spectral_bands:
            logger.info("Step 2.5: Adding EVI using eemont...")
            satellite_type = config.get('I_satellite', 'Sentinel')
            # Map satellite type to eemont format
            eemont_satellite = {
                'Sentinel': 'Sentinel',
                'Landsat': 'Landsat'
            }.get(satellite_type, 'Sentinel')
            
            collection_with_indices = collection_with_indices.spectralIndices(
                index=['EVI'],
                satellite_type=eemont_satellite,
                G=2.5,  # EVI gain factor
                C1=6.0,  # EVI coefficient 1
                C2=7.5,  # EVI coefficient 2
                drop=False  # Keep original bands
            )
            logger.info("✅ EVI added using eemont")
    except ImportError:
        logger.warning("eemont not available. EVI will not be added.")
    except Exception as e:
        logger.warning(f"Failed to add EVI using eemont: {e}")
    
    # Step 3: Apply smoothing
    
    # Verify that requested bands exist in the collection
    first_img = collection_with_indices.first()
    available_bands = first_img.bandNames().getInfo()
    valid_bands = [b for b in spectral_bands if b in available_bands]
    
    if not valid_bands:
        logger.warning(f"None of the requested bands {spectral_bands} are available!")
        logger.warning(f"Available bands: {available_bands}")
        logger.warning("Skipping smoothing step.")
        return collection_with_indices
    
    if len(valid_bands) < len(spectral_bands):
        missing = set(spectral_bands) - set(valid_bands)
        logger.warning(f"Some requested bands are not available: {missing}")
        logger.info(f"Will smooth only: {valid_bands}")
    
    # Step 3: Remove outliers (drops and spikes) - following 02b workflow
    logger.info(f"Step 3: Removing outliers using rolling median...")
    logger.info(f"   Outlier window: {outlier_window} images")
    logger.info(f"   Outlier threshold: {outlier_threshold * 100}%")
    logger.info(f"   Bands to process: {valid_bands}")

    collection_no_outliers = remove_drops_and_spikes_gee(
        image_collection=collection_with_indices,
        band_names=valid_bands,
        window_size=outlier_window,
        threshold_percent=outlier_threshold
    )
    logger.info("✅ Outlier removal complete!")

    # Step 4: Interpolate temporal gaps - following 02b workflow
    logger.info(f"Step 4: Interpolating gaps in time series...")
    logger.info(f"   This fills missing dates/intervals, similar to XArray np.interp()")
    
    collection_interpolated = interpolate_temporal_gaps(
        collection_no_outliers,
        band_names=valid_bands
    )
    logger.info("✅ Gap interpolation complete!")
    
    # Step 5: Apply Savitzky-Golay smoothing - following 02b workflow
    logger.info(f"Step 5: Applying Savitzky-Golay filtering...")
    logger.info(f"   Window length: {smoothing_window}")
    logger.info(f"   Polynomial order: {smoothing_polyorder}")
    logger.info(f"   Bands to filter: {valid_bands}")
    
    collection_with_sg = savgol_filter(
        collection_interpolated,
        band_names=valid_bands,
        window_length=smoothing_window,
        polyorder=smoothing_polyorder
    )
    
    # IMPORTANT: Explicitly select only the bands we need (following 02b)
    collection_with_sg = collection_with_sg.map(
        lambda img: img.select(valid_bands)
    )
    
    logger.info("✅ Savitzky-Golay filtering complete!")
    logger.info(f"   Filtered bands: {valid_bands}")
    
    return collection_with_sg


def savgol_filter(
    image_collection: ee.ImageCollection,
    band_names: List[str] = ['NDVI', 'EVI'],
    window_length: int = 7,
    polyorder: int = 2
) -> ee.ImageCollection:
    """
    Apply Savitzky-Golay filtering (smoothing) to spectral indices.
    
    WHAT IT DOES:
    - Smooths time series using weighted temporal averaging (Gaussian weights)
    - Preserves existing valid pixels (applies temporal smoothing to reduce noise)
    
    WHAT IT DOES NOT DO:
    - Does NOT interpolate missing values
    - Does NOT fill gaps where data is missing (e.g., where cloudM=0)
    - Does NOT create new data points
    
    The input collection should already have cloud masking applied (e.g., via OSI).
    
    Parameters
    ----------
    image_collection : ee.ImageCollection
        Input image collection with spectral indices bands
    band_names : list
        List of band names to filter (default: ['NDVI', 'EVI'])
    window_length : int
        Window length (must be odd, >= polyorder+1). Default: 7
    polyorder : int
        Polynomial order (typically 2-3). Default: 2
    
    Returns
    -------
    ee.ImageCollection
        Filtered collection with smoothed bands
    """
    # Validate window_length
    if window_length % 2 == 0:
        window_length += 1
    if window_length < polyorder + 1:
        window_length = polyorder + 1
        if window_length % 2 == 0:
            window_length += 1
    
    half_window = (window_length - 1) // 2
    collection_list = image_collection.toList(image_collection.size())
    n_images = image_collection.size().getInfo()
    max_idx_ee = ee.Number(n_images - 1)
    half_window_ee = ee.Number(half_window)
    
    logger.info(f"Applying Savitzky-Golay filtering...")
    logger.info(f"   Window length: {window_length}")
    logger.info(f"   Polynomial order: {polyorder}")
    logger.info(f"   Bands to filter: {band_names}")
    
    def process_image(i):
        center_idx = ee.Number(i).int()
        center_img = ee.Image(collection_list.get(center_idx))
        result_img = center_img
        
        for band_name in band_names:
            center_band = center_img.select([band_name])
            
            # Calculate window bounds with boundary safety
            start_idx = center_idx.subtract(half_window_ee).max(0)
            end_idx = center_idx.add(half_window_ee).min(max_idx_ee)
            
            # Get window images
            window_indices = ee.List.sequence(start_idx, end_idx)
            window_images = window_indices.map(
                lambda idx: ee.Image(collection_list.get(ee.Number(idx)))
            )
            window_collection = ee.ImageCollection.fromImages(window_images)
            
            # Get window bands
            window_bands = window_collection.select([band_name])
            
            # Calculate weights (Gaussian-like, centered at center_idx)
            def get_weight(img_idx):
                idx_num = ee.Number(img_idx)
                distance = idx_num.subtract(center_idx).abs()
                # Gaussian weight: exp(-0.5 * (distance/sigma)^2)
                # Use half_window as sigma
                sigma = half_window_ee.max(1)  # Avoid division by zero
                weight = ee.Number(1).subtract(
                    distance.divide(sigma).pow(2).multiply(0.5)
                ).exp()
                return weight
            
            # Apply weights and calculate weighted mean
            weighted_sum = ee.Image(0)
            weight_sum = ee.Number(0)
            
            def accumulate_weighted(img, accum):
                accum = ee.Dictionary(accum)
                img_idx = ee.Number(accum.get('idx'))
                weighted_sum = ee.Image(accum.get('weighted_sum'))
                weight_sum = ee.Number(accum.get('weight_sum'))
                
                band = img.select([band_name])
                weight = get_weight(img_idx)
                
                weighted_sum = weighted_sum.add(band.multiply(weight))
                weight_sum = weight_sum.add(weight)
                
                return ee.Dictionary({
                    'idx': img_idx.add(1),
                    'weighted_sum': weighted_sum,
                    'weight_sum': weight_sum
                })
            
            initial = ee.Dictionary({
                'idx': start_idx,
                'weighted_sum': ee.Image(0),
                'weight_sum': ee.Number(0)
            })
            
            result = window_collection.iterate(accumulate_weighted, initial)
            weighted_sum = ee.Image(ee.Dictionary(result).get('weighted_sum'))
            weight_sum = ee.Number(ee.Dictionary(result).get('weight_sum'))
            
            # Normalize by sum of weights
            smoothed_band = weighted_sum.divide(weight_sum)
            
            # Preserve mask from original band
            smoothed_band = smoothed_band.updateMask(center_band.mask())
            
            result_img = result_img.addBands(smoothed_band.rename([band_name]), None, True)
        
        return result_img
    
    processed = ee.List.sequence(0, n_images - 1).map(process_image)
    result_collection = ee.ImageCollection.fromImages(processed)
    
    logger.info("✅ Savitzky-Golay filtering complete!")
    
    return result_collection


def extract_time_series_for_visualization(
    collection_before: ee.ImageCollection,
    collection_after: ee.ImageCollection,
    sample_point: ee.Geometry,
    bands: List[str],
    scale: float = 10,
    max_pixels: int = 1e10,
    best_effort: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract point-based time series from two EE ImageCollections (before/after processing).
    
    This function extracts time series data from two ImageCollections at a sample point,
    typically used to compare original vs processed (e.g., smoothed) data for visualization.
    
    **Note:** This function requires that images in the collections have `system:time_start`
    property set. Monthly composites created with `create_monthly_composites()` already
    have this property set.
    
    Parameters
    ----------
    collection_before : ee.ImageCollection
        Original collection (e.g., before Savitzky-Golay smoothing).
        Must have `system:time_start` property on each image.
    collection_after : ee.ImageCollection
        Processed collection (e.g., after Savitzky-Golay smoothing).
        Must have `system:time_start` property on each image.
    sample_point : ee.Geometry
        Point geometry at which to sample the time series (e.g., ee.Geometry.Point([lon, lat])).
    bands : list[str]
        Band names to extract (must exist in both collections), e.g., ['NDVI', 'EVI'].
    scale : float, optional
        Scale in meters for reduceRegion (default: 10).
    max_pixels : int, optional
        maxPixels for reduceRegion (default: 1e10).
    best_effort : bool, optional
        bestEffort flag for reduceRegion (default: True).
    
    Returns
    -------
    tuple
        (df_before, df_after) - Two pandas DataFrames with columns ['date', *bands],
        sorted chronologically and ready for plotting.
    
    Raises
    ------
    ValueError
        If images in the collection don't have `system:time_start` property.
    
    Examples
    --------
    >>> sample_point = ee.Geometry.Point([111.81746, -0.41587])
    >>> bands = ['NDVI', 'EVI']
    >>> df_before, df_after = extract_time_series_for_visualization(
    ...     collection_before=collection_with_indices,
    ...     collection_after=collection_with_sg,
    ...     sample_point=sample_point,
    ...     bands=bands,
    ...     scale=10
    ... )
    """
    logger.info("=" * 60)
    logger.info("Extracting time series for visualization")
    logger.info("=" * 60)
    
    # Check if first image has system:time_start property
    def _check_time_start(collection: ee.ImageCollection, label: str) -> None:
        """Check if collection has system:time_start property."""
        try:
            first_img = collection.first()
            time_start = first_img.get('system:time_start').getInfo()
            if time_start is None:
                raise ValueError(
                    f"{label} collection images do not have 'system:time_start' property. "
                    f"Monthly composites should have this property set. "
                    f"Please ensure images are created with create_monthly_composites() "
                    f"or have system:time_start set manually."
                )
            logger.info(f"   ✅ {label} collection has system:time_start property")
        except Exception as e:
            if "does not have a 'system:time_start' property" in str(e) or "system:time_start" in str(e):
                raise ValueError(
                    f"{label} collection images do not have 'system:time_start' property. "
                    f"Monthly composites should have this property set. "
                    f"Please ensure images are created with create_monthly_composites() "
                    f"or have system:time_start set manually."
                ) from e
            raise
    
    _check_time_start(collection_before, "BEFORE")
    _check_time_start(collection_after, "AFTER")
    
    def _extract_point_timeseries(collection: ee.ImageCollection, label: str) -> pd.DataFrame:
        """
        Extract point-based time series from a single ImageCollection.
        
        Parameters
        ----------
        collection : ee.ImageCollection
            Collection to extract from (must have system:time_start property).
        label : str
            Label for logging (e.g., "BEFORE" or "AFTER").
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns ['date', *bands].
        """
        logger.info(f"   Extracting {label} time series...")
        
        def reducer(img: ee.Image) -> ee.Feature:
            """Reduce image to point statistics."""
            img_selected = img.select(bands)
            stats = img_selected.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=sample_point,
                scale=scale,
                bestEffort=best_effort,
                maxPixels=max_pixels,
            )
            
            # Get date from image (requires system:time_start property)
            date_str = img.date().format('YYYY-MM-dd')
            props = {'date': date_str}
            for band in bands:
                props[band] = stats.get(band)
            return ee.Feature(sample_point, props)
        
        features = collection.map(reducer)
        rows = features.getInfo()['features']
        data = []
        
        for feat in rows:
            props = feat['properties']
            row = {'date': props.get('date')}
            for band in bands:
                row[band] = props.get(band)
            data.append(row)
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        df = df.replace([None, -9999], np.nan)
        
        logger.info(f"   ✅ Converted {label}: {len(df)} time steps")
        return df
    
    # Log extraction parameters
    point_coords = sample_point.getInfo()['coordinates']
    logger.info(f"   Sample point: {point_coords}")
    logger.info(f"   Bands to compare: {', '.join(bands)}")
    
    logger.info("\n   Comparison:")
    logger.info("   1. BEFORE: original collection")
    logger.info("   2. AFTER:  processed collection (e.g. SG smoothed)")
    
    # Extract time series from both collections
    df_before = _extract_point_timeseries(
        collection_before,
        "BEFORE (original, non-smoothed)"
    )
    df_after = _extract_point_timeseries(
        collection_after,
        "AFTER (smoothed)"
    )
    
    logger.info(f"\n   Before: {len(df_before)} time steps")
    logger.info(f"   After: {len(df_after)} time steps")
    if len(df_before) > 0 and len(df_after) > 0:
        logger.info(f"   Date range: {df_before['date'].min()} to {df_before['date'].max()}")
    
    logger.info("\n💾 DataFrames ready for plotting:")
    logger.info("   - df_before: Original values (before processing)")
    logger.info("   - df_after:  Processed values (after processing)")
    
    return df_before, df_after


def plot_sg_comparison(
    ts_before_df: pd.DataFrame,
    ts_after_df: pd.DataFrame,
    bands: List[str],
    seed: int = 42
) -> None:
    """
    Plot before/after Savitzky-Golay time series comparison for a set of bands.
    
    This function creates side-by-side plots comparing original vs smoothed time series
    and prints summary statistics including noise reduction metrics.
    
    Parameters
    ----------
    ts_before_df : pd.DataFrame
        DataFrame from the original collection; must include a 'date' column and band columns.
    ts_after_df : pd.DataFrame
        DataFrame from the smoothed collection; same structure as ts_before_df.
    bands : list[str]
        Band names to compare (e.g., ['NDVI', 'EVI']).
    seed : int, optional
        Random seed for reproducible color assignments (default: 42).
    
    Returns
    -------
    None
        Displays plots and prints statistics to console.
    
    Examples
    --------
    >>> bands = ['NDVI', 'EVI']
    >>> plot_sg_comparison(ts_before_df, ts_after_df, bands)
    """
    if not HAS_MATPLOTLIB:
        logger.error("matplotlib is not available. Cannot create plots.")
        return
    
    bands_to_plot = bands
    n_bands = len(bands_to_plot)
    
    if n_bands == 0:
        logger.warning("⚠️  No bands supplied to plot_sg_comparison!")
        return
    
    logger.info("=" * 60)
    logger.info("Creating Savitzky-Golay comparison plots")
    logger.info("=" * 60)
    
    # Generate reproducible color pairs for each band
    np.random.seed(seed)
    colors_info = {}
    for band in bands_to_plot:
        rgb_before = np.random.randint(100, 255, size=3)
        rgb_after = (rgb_before * 0.4).astype(int)
        colors_info[band] = {
            'before': '#{:02x}{:02x}{:02x}'.format(*rgb_before),
            'after': '#{:02x}{:02x}{:02x}'.format(*rgb_after)
        }
    
    # Print selected color scheme
    logger.info("🎨 Color Scheme for Visualization:")
    logger.info("=" * 60)
    for band, colors in colors_info.items():
        logger.info(f"  {band}:")
        logger.info(f"    Before SG: {colors['before']}")
        logger.info(f"    After SG:  {colors['after']}")
    logger.info("")
    
    # Create figure with one subplot per band
    fig, axes = plt.subplots(n_bands, 1, figsize=(14, 5 * n_bands))
    if n_bands == 1:
        axes = [axes]
    
    fig.suptitle('Savitzky-Golay Filtering: Before vs After Comparison', 
                 fontsize=16, fontweight='bold')
    
    for ax, band in zip(axes, bands_to_plot):
        if band in ts_before_df.columns and band in ts_after_df.columns:
            color_before = colors_info[band]['before']
            color_after = colors_info[band]['after']
            
            ax.plot(ts_before_df['date'], ts_before_df[band],
                   'o-', color=color_before, alpha=0.6, linewidth=1.5, markersize=4,
                   label='Before SG (Original)', zorder=1)
            ax.plot(ts_after_df['date'], ts_after_df[band],
                   '-', color=color_after, linewidth=2.5,
                   label='After SG (Smoothed)', zorder=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(band, fontsize=12)
        ax.set_title(f'{band} Time Series', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    logger.info("\n📊 Summary Statistics:")
    logger.info("=" * 60)
    for band in bands_to_plot:
        if band in ts_before_df.columns and band in ts_after_df.columns:
            before_mean = ts_before_df[band].mean()
            before_std = ts_before_df[band].std()
            after_mean = ts_after_df[band].mean()
            after_std = ts_after_df[band].std()
            
            logger.info(f"\n{band} Statistics:")
            logger.info(f"  Before SG - Mean: {before_mean:.4f}, Std: {before_std:.4f}")
            logger.info(f"  After SG  - Mean: {after_mean:.4f}, Std: {after_std:.4f}")
            if before_std > 0:
                noise_reduction = (1 - after_std / before_std) * 100
                logger.info(f"  Noise Reduction: {noise_reduction:.1f}%")
    
    logger.info("=" * 60)

#### INTERPOLATION SMOOTHING
def fill_temporal_gaps_linear(collection, bands):
    """Linearly interpolate masked pixels; fall back to nearest neighbor only at the edges."""
    import ee

    collection = collection.sort('system:time_start')
    img_list = collection.toList(collection.size())

    def augment(image, time_band):
        image = ee.Image(image)
        data = image.select(bands).toDouble()
        time_value = ee.Image.constant(ee.Number(image.get('system:time_start'))).rename(time_band).toDouble()
        time_mask = data.mask().reduce(ee.Reducer.max())
        time_image = time_value.updateMask(time_mask)
        combined = ee.Image(data.addBands(time_image))
        combined = ee.Image(combined.copyProperties(image, image.propertyNames()))
        return combined

    first_forward = augment(img_list.get(0), 'time_prev')

    def forward_iter(current, prev_list):
        prev_list = ee.List(prev_list)
        prev_img = ee.Image(prev_list.get(-1))
        current_img = augment(current, 'time_prev')
        filled = ee.Image(current_img.unmask(prev_img))
        filled = ee.Image(filled.copyProperties(current_img, current_img.propertyNames()))
        return prev_list.add(filled)

    forward_list = ee.List(
        img_list.slice(1).iterate(forward_iter, ee.List([first_forward]))
    )

    reversed_list = img_list.reverse()
    first_backward = augment(reversed_list.get(0), 'time_next')

    def backward_iter(current, prev_list):
        prev_list = ee.List(prev_list)
        prev_img = ee.Image(prev_list.get(-1))
        current_img = augment(current, 'time_next')
        filled = ee.Image(current_img.unmask(prev_img))
        filled = ee.Image(filled.copyProperties(current_img, current_img.propertyNames()))
        return prev_list.add(filled)

    backward_list = ee.List(
        reversed_list.slice(1).iterate(backward_iter, ee.List([first_backward]))
    ).reverse()

    def interpolate(idx, acc):
        acc = ee.List(acc)
        idx = ee.Number(idx)

        original = ee.Image(img_list.get(idx))
        prev_img = ee.Image(forward_list.get(idx))
        next_img = ee.Image(backward_list.get(idx))

        prev_values = prev_img.select(bands)
        next_values = next_img.select(bands)
        prev_time = prev_img.select('time_prev')
        next_time = next_img.select('time_next')
        current_time = ee.Image.constant(ee.Number(original.get('system:time_start'))).toDouble()

        time_diff = next_time.subtract(prev_time)
        interp_mask = prev_time.mask().And(next_time.mask()).And(time_diff.neq(0))

        fraction = current_time.subtract(prev_time).divide(time_diff)
        fraction = fraction.updateMask(interp_mask).clamp(0, 1)

        interpolated = prev_values.add(
            next_values.subtract(prev_values).multiply(fraction)
        ).updateMask(interp_mask)

        filled = original.select(bands)
        filled = filled.unmask(interpolated)
        filled = filled.unmask(prev_values)
        filled = filled.unmask(next_values)

        updated = original.addBands(filled, overwrite=True)
        updated = ee.Image(updated.copyProperties(original, original.propertyNames()))
        return acc.add(updated)

    filled_list = ee.List(
        ee.List.sequence(0, collection.size().subtract(1)).iterate(interpolate, ee.List([]))
    )

    return ee.ImageCollection(filled_list).sort('system:time_start')
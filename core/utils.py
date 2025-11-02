"""
Utility Functions Module

This module contains utility functions for data coverage diagnosis,
report generation, and other helper functions.
"""

import os
import numpy as np
import geopandas as gpd
import xarray as xr  # Required dependency for STAC processing
import logging
from shapely.geometry import box
from typing import Optional, Dict, Any, List, Union
import tempfile

# GEE imports (optional)
try:
    import ee
    import geemap
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False

logger = logging.getLogger(__name__)


def refresh_planetary_computer_token():
    """
    Refresh Planetary Computer authentication token with more aggressive approach
    
    Returns:
        bool: True if refresh was successful, False otherwise
    """
    try:
        import planetary_computer
        import pystac_client
        import time
        
        # Force token refresh by creating a new client
        print("🔄 Refreshing Planetary Computer token...")
        
        # Add a small delay to ensure token refresh
        time.sleep(1)
        
        # Test with a simple request
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        # Try to get a collection to test the connection
        catalog.get_collection("sentinel-2-l2a")
        print("✅ Token refresh successful!")
        return True
        
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        return False

def create_fresh_stac_client(config):
    """
    Create a completely fresh STAC client with new authentication
    
    Args:
        config: Configuration dictionary
        
    Returns:
        pystac_client.Client: Fresh STAC client
    """
    import pystac_client
    import planetary_computer
    import time
    
    print("🔄 Creating fresh STAC client...")
    
    # Add delay to ensure fresh authentication
    time.sleep(2)
    
    # Create new client with fresh modifier
    catalog = pystac_client.Client.open(
        config['url_satellite_cloud'],
        modifier=planetary_computer.sign_inplace,
    )
    
    return catalog


class DataUtils:
    """
    Utility functions for data processing and analysis.
    """
    
    def __init__(self, config_manager, use_gee: bool = True):
        """
        Initialize the data utilities.
        
        Args:
            config_manager: Configuration manager instance or config dictionary
                - If ConfigManager: uses config_manager.config
                - If dict: uses directly
            use_gee (bool): Whether to initialize Google Earth Engine. Default is True.
                            If True and GEE_AVAILABLE, will initialize GEE following main.py pattern.
        """
        # Handle both ConfigManager instance and dict
        if hasattr(config_manager, 'config'):
            # It's a ConfigManager instance
            self.config = config_manager.config
            self.config_manager = config_manager
        elif isinstance(config_manager, dict):
            # It's a config dictionary
            self.config = config_manager
            self.config_manager = None
        else:
            raise TypeError(f"config_manager must be ConfigManager instance or dict, got {type(config_manager)}")
        
        self.use_gee = use_gee
        self._gee_initialized = False
        
        # Initialize GEE if requested and available
        if use_gee:
            self._initialize_gee()
    
    def _is_gee_initialized(self) -> bool:
        """
        Check if Google Earth Engine is initialized.
        
        This is a safe replacement for ee.data._initialized which is no longer
        available in newer versions of the Earth Engine API.
        
        Returns:
            bool: True if GEE is initialized, False otherwise.
        """
        if not GEE_AVAILABLE:
            return False
        
        try:
            import ee
            # Try to access a method that definitely requires initialization
            # getAssetRoots() is a lightweight operation that requires auth
            try:
                # This will raise an exception if not initialized
                ee.data.getAssetRoots()
                return True
            except ee.EEException as e:
                # Check if the error is about not being initialized
                error_msg = str(e).lower()
                if 'not initialized' in error_msg or 'please authenticate' in error_msg:
                    return False
                # Other EEException might indicate initialization but other issues
                # Return True since GEE seems to be initialized (just has another problem)
                return True
            except (RuntimeError, AttributeError, Exception) as e:
                # For non-EE exceptions, check error message
                error_msg = str(e).lower()
                if 'not initialized' in error_msg or 'please authenticate' in error_msg or 'initialize' in error_msg:
                    return False
                # Unknown error - assume not initialized to be safe
                return False
        except Exception:
            return False
    
    def _initialize_gee(self):
        """
        Initialize Google Earth Engine following the same pattern as main.py
        
        Tries gee_lib auth first (container environment), then falls back to direct initialization.
        """
        if not GEE_AVAILABLE:
            logger.warning("GEE libraries not available. Skipping GEE initialization.")
            return
        
        try:
            import ee
            from dotenv import load_dotenv
            
            # Load environment variables
            load_dotenv()
            
            # Check if already initialized
            if self._is_gee_initialized():
                logger.info("GEE already initialized")
                self._gee_initialized = True
                return
            
            logger.info("Initializing Google Earth Engine...")
            
            # Try to import gee_lib auth if available (container environment)
            try:
                # Try gee_lib auth (container environment)
                try:
                    from gee_lib.osi.auth import initialize_gee
                    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
                    if project_id:
                        initialize_gee(project_id=project_id, use_service_account=True)
                        logger.info("GEE initialized successfully using gee_lib auth")
                        self._gee_initialized = True
                        return
                    else:
                        logger.warning("GOOGLE_CLOUD_PROJECT not set, falling back to direct initialization")
                except ImportError:
                    logger.debug("gee_lib.osi.auth not available, trying direct initialization")
                
                # Fallback to direct GEE initialization
                try:
                    ee.Initialize()
                    logger.info("GEE initialized directly")
                    self._gee_initialized = True
                except Exception as init_error:
                    logger.error(f"Failed to initialize GEE directly: {init_error}")
                    raise RuntimeError(f"Failed to initialize GEE: {init_error}")
                    
            except Exception as e:
                logger.warning(f"Could not use gee_lib auth: {e}. Attempting direct GEE initialization...")
                # Final fallback to direct initialization
                try:
                    ee.Initialize()
                    logger.info("GEE initialized directly (fallback)")
                    self._gee_initialized = True
                except Exception as init_error:
                    logger.error(f"Failed to initialize GEE (fallback): {init_error}")
                    raise RuntimeError(f"Failed to initialize GEE: {init_error}")
        
        except ImportError:
            raise ImportError("Earth Engine library not available. Install with: pip install earthengine-api geemap")
    
    def load_geodataframe(self, input_json: str) -> gpd.GeoDataFrame:
        """
        Load GeoDataFrame from JSON file.
        
        Args:
            input_json (str): Path to the input JSON/GeoJSON file
            
        Returns:
            gpd.GeoDataFrame: Loaded geodataframe
        """
        try:
            gdf = gpd.read_file(input_json)
            logger.info(f"GeoDataFrame loaded from {input_json}")
            logger.info(f"Bounding box: {gdf.total_bounds}")
            return gdf
        except Exception as e:
            logger.error(f"Error loading GeoDataFrame: {e}")
            raise
    
    def load_geodataframe_gee(self, input_json: str, auto_init_gee: bool = True) -> tuple:
        """
        Load GeoDataFrame and convert to Earth Engine format.
        
        Args:
            input_json (str): Path to the input JSON/GeoJSON file
            auto_init_gee (bool): If True and GEE not initialized, will auto-initialize GEE.
                                 Default is True.
            
        Returns:
            tuple: (gpd.GeoDataFrame, ee.FeatureCollection) - Loaded geodataframe and EE FeatureCollection
        """
        if not GEE_AVAILABLE:
            raise ImportError("GEE library not available. Install with: pip install earthengine-api geemap")
        
        import ee
        
        # Check if Earth Engine is initialized, auto-initialize if needed
        if not self._is_gee_initialized():
            if auto_init_gee:
                logger.info("GEE not initialized. Auto-initializing...")
                self._initialize_gee()
            else:
                raise RuntimeError(
                    "Earth Engine client library not initialized. "
                    "Initialize DataUtils with use_gee=True or call _initialize_gee() manually. "
                    "See http://goo.gle/ee-auth."
                )
        
        # Verify initialization was successful
        if not self._is_gee_initialized():
            raise RuntimeError("Failed to initialize Earth Engine. Please check authentication.")
        
        # Load GeoDataFrame using existing method
        gdf = self.load_geodataframe(input_json)
        
        try:
            # Check if input is GeoJSON and convert to temporary shapefile for OSI compatibility
            if input_json.lower().endswith(('.json', '.geojson')):
                # Create temporary shapefile
                temp_dir = tempfile.mkdtemp()
                temp_shp_path = os.path.join(temp_dir, 'aoi_temp.shp')
                
                # Add 'id' column as required by OSI (OID field)
                gdf_with_id = gdf.copy()
                gdf_with_id['id'] = range(len(gdf_with_id))
                
                # Save as shapefile
                gdf_with_id.to_file(temp_shp_path)
                
                # Convert shapefile to Earth Engine format
                aoi_ee = geemap.shp_to_ee(temp_shp_path)
                
                logger.info(f"AOI converted from GeoJSON to temporary shapefile: {temp_shp_path}")
                logger.info("Added 'id' column as OID field for OSI compatibility")
                
                return gdf, aoi_ee
            else:
                # Direct conversion for shapefile inputs
                aoi_ee = geemap.shp_to_ee(input_json)
                logger.info("AOI converted to Earth Engine format")
                
                return gdf, aoi_ee
                
        except Exception as e:
            logger.error(f"Failed to convert AOI to Earth Engine format: {e}")
            raise
    
    def diagnose_data_coverage(self, data: xr.Dataset, bbox: box) -> Dict[str, Any]:
        """
        Diagnose data coverage issues - check AOI size vs actual data coverage.
        
        Args:
            data (xr.Dataset): Input dataset (xarray.Dataset)
            bbox (box): Bounding box of the AOI
            
        Returns:
            Dict[str, Any]: Coverage diagnosis report
        """
        logger.info("🔍 Diagnosing data coverage issues...")
        
        # Calculate AOI area
        aoi_area_km2 = bbox.area * 111 * 111  # Rough conversion to km² (1 degree ≈ 111 km)
        
        # Get data extent
        x_coords = data.x.values
        y_coords = data.y.values
        data_xmin, data_xmax = x_coords.min(), x_coords.max()
        data_ymin, data_ymax = y_coords.min(), y_coords.max()
        
        # Calculate data area
        data_area_km2 = (data_xmax - data_xmin) * (data_ymax - data_ymin) * 111 * 111
        
        # Calculate pixel size and total pixels
        pixel_size_x = (data_xmax - data_xmin) / (len(x_coords) - 1)
        pixel_size_y = (data_ymax - data_ymin) / (len(y_coords) - 1)
        total_pixels = len(x_coords) * len(y_coords)
        
        # Check coverage for each band (use renamed bands if available, fallback to original)
        coverage_stats = {}
        # Define key bands to check coverage for
        key_bands = ['blue', 'green', 'red', 'nir']  # Renamed bands
        fallback_bands = ['B02', 'B03', 'B04', 'B08']  # Original bands as fallback
        
        # Try renamed bands first, then fallback to original
        bands_to_check = []
        for band in key_bands:
            if band in data.data_vars:
                bands_to_check.append(band)
            else:
                # Find corresponding original band
                original_band = None
                if 'band_mapping' in self.config.config:
                    for orig, renamed in self.config.config['band_mapping'].items():
                        if renamed == band and orig in data.data_vars:
                            original_band = orig
                            break
                if original_band:
                    bands_to_check.append(original_band)
        
        # If no renamed bands found, use fallback
        if not bands_to_check:
            bands_to_check = [band for band in fallback_bands if band in data.data_vars]
        
        for band in bands_to_check:
            if band in data.data_vars:
                band_data = data[band].isel(time=0)
                valid_pixels = (~np.isnan(band_data)).sum().values
                coverage_percent = (valid_pixels / total_pixels * 100)
                coverage_stats[band] = {
                    'valid_pixels': int(valid_pixels),
                    'total_pixels': int(total_pixels),
                    'coverage_percent': float(coverage_percent)
                }
        
        # Check if validity mask exists
        has_validity_mask = 'is_valid' in data.data_vars
        if has_validity_mask:
            valid_mask = data['is_valid'].isel(time=0)
            valid_pixels_mask = valid_mask.sum().values
            coverage_percent_mask = (valid_pixels_mask / total_pixels * 100)
        else:
            valid_pixels_mask = 0
            coverage_percent_mask = 0
        
        # Calculate actual covered area
        avg_coverage = np.mean([stats['coverage_percent'] for stats in coverage_stats.values()])
        actual_covered_area_km2 = data_area_km2 * (avg_coverage / 100)
        
        # Diagnosis
        diagnosis = {
            'aoi_analysis': {
                'aoi_area_km2': float(aoi_area_km2),
                'aoi_bounds': bbox.bounds,
                'aoi_width_km': float((bbox.bounds[2] - bbox.bounds[0]) * 111),
                'aoi_height_km': float((bbox.bounds[3] - bbox.bounds[1]) * 111)
            },
            'data_analysis': {
                'data_area_km2': float(data_area_km2),
                'data_bounds': (float(data_xmin), float(data_ymin), float(data_xmax), float(data_ymax)),
                'data_width_km': float((data_xmax - data_xmin) * 111),
                'data_height_km': float((data_ymax - data_ymin) * 111),
                'pixel_size_x_m': float(pixel_size_x * 111000),  # Convert to meters
                'pixel_size_y_m': float(pixel_size_y * 111000),
                'total_pixels': int(total_pixels)
            },
            'coverage_analysis': {
                'band_coverage': coverage_stats,
                'validity_mask_coverage_percent': float(coverage_percent_mask),
                'average_coverage_percent': float(avg_coverage),
                'actual_covered_area_km2': float(actual_covered_area_km2)
            },
            'diagnosis': {
                'aoi_vs_data_ratio': float(aoi_area_km2 / data_area_km2) if data_area_km2 > 0 else 0,
                'coverage_quality': 'excellent' if avg_coverage > 80 else 'good' if avg_coverage > 50 else 'poor' if avg_coverage > 20 else 'very_poor',
                'potential_issues': []
            }
        }
        
        # Identify potential issues
        if avg_coverage < 20:
            diagnosis['diagnosis']['potential_issues'].append("Very low data coverage - most pixels are empty")
        if avg_coverage < 50:
            diagnosis['diagnosis']['potential_issues'].append("Low data coverage - significant gaps in data")
        if aoi_area_km2 > data_area_km2 * 2:
            diagnosis['diagnosis']['potential_issues'].append("AOI is much larger than data extent - consider smaller AOI")
        if pixel_size_x * 111000 > 20:  # More than 20m pixels
            diagnosis['diagnosis']['potential_issues'].append("Large pixel size - data might be too coarse")
        if not has_validity_mask:
            diagnosis['diagnosis']['potential_issues'].append("No validity mask - cloud masking might not be applied")
        
        # Recommendations
        recommendations = []
        if avg_coverage < 30:
            recommendations.extend([
                "Increase cloud cover threshold (e.g., 80% instead of 60%)",
                "Expand date range to get more images",
                "Consider using different satellite collection (Landsat, MODIS)",
                "Reduce AOI size to focus on areas with better coverage"
            ])
        if aoi_area_km2 > 1000:  # Large AOI
            recommendations.append("Consider splitting large AOI into smaller chunks")
        
        diagnosis['recommendations'] = recommendations
        
        # Print summary
        print("🔍 DATA COVERAGE DIAGNOSIS")
        print("=" * 50)
        print(f"📍 AOI Area: {aoi_area_km2:.1f} km²")
        print(f"📊 Data Area: {data_area_km2:.1f} km²")
        print(f"📈 Average Coverage: {avg_coverage:.1f}%")
        print(f"🎯 Actual Covered Area: {actual_covered_area_km2:.1f} km²")
        print(f"📐 Pixel Size: {pixel_size_x * 111000:.1f}m x {pixel_size_y * 111000:.1f}m")
        print(f"🏷️ Coverage Quality: {diagnosis['diagnosis']['coverage_quality']}")
        
        if diagnosis['diagnosis']['potential_issues']:
            print(f"\n⚠️ POTENTIAL ISSUES:")
            for issue in diagnosis['diagnosis']['potential_issues']:
                print(f"   - {issue}")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   - {rec}")
        
        return diagnosis
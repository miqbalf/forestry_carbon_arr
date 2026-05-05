"""
STAC-based Satellite Data Processing Module

This module handles satellite data acquisition, processing, and storage using
the STAC (SpatioTemporal Asset Catalog) approach with Planetary Computer.
"""

import os
import time
import numpy as np
import geopandas as gpd
import xarray as xr
import stackstac
import pystac_client
import planetary_computer
import dask.diagnostics
from shapely.geometry import box
from shapely import to_geojson
from typing import Optional, Dict, Any, Union, List
import logging
import rasterio
from rasterio.errors import RasterioIOError
import dask
from dask.diagnostics import ProgressBar
from collections import defaultdict
import gcsfs

logger = logging.getLogger(__name__)


class STACProcessor:
    """
    Handles STAC-based satellite data acquisition and processing.
    """
    
    def __init__(self, config_manager, datetime_override: Optional[Union[str, List[str]]] = None,
                 resolution_override: Optional[int] = None):
        """
        Initialize the STAC processor.
        
        Args:
            config_manager: Configuration manager instance
            datetime_override: Optional datetime override for STAC search (can override config)
            resolution_override: Optional resolution override in meters (can override config)
        """
        self.config = config_manager
        self.satellite_config = config_manager.get_stac_config(
            datetime_override=datetime_override,
            resolution_override=resolution_override
        )
    
    def search_satellite_data(self, bbox: box, datetime_range: Optional[str] = None) -> list:
        """
        Search for satellite data using STAC.
        
        Args:
            bbox (box): Bounding box for search
            datetime_range (Optional[str]): Date range for search (e.g., "2017-01-01/2024-12-31").
                If None, uses datetime from config.
            
        Returns:
            list: List of STAC items found
            
        Raises:
            ValueError: If no satellite images found
        """
        logger.info("Searching for satellite data using STAC...")
        
        # Use provided datetime_range or fall back to config
        datetime_param = datetime_range if datetime_range is not None else self.satellite_config['datetime']
        logger.info(f"Using date range: {datetime_param}")
                
        # STAC Search
        catalog = pystac_client.Client.open(
            self.satellite_config['url'],
            modifier=planetary_computer.sign_inplace,
        )
        
        search = catalog.search(
            collections=[self.satellite_config['collection']],
            intersects=to_geojson(bbox),
            datetime=datetime_param,
            query={"eo:cloud_cover": {"lt": self.satellite_config['cloud_cover']}},
        )
        
        # Get items
        # Check how many items were returned
        items = search.item_collection()
        logger.info(f"Found {len(items)} satellite images")
        
        if len(items) == 0:
            raise ValueError("No satellite images found for the given criteria")

        items_stac = search.item_collection()
        logger.info(f"STAC found: {len(items_stac)} total images before filtering duplicates")

        # --- MODIFICATION START ---

        # Group items by their unique acquisition identifier (the scene ID part)
        # Example ID: S2A_MSIL2A_20240607T075611_R035_T36NVH_20240607T134827
        # The unique scene part is S2A_MSIL2A_20240607T075611_R035_T36NVH
        grouped_items = defaultdict(list)
        for item in items_stac:
            # Split the ID at the final underscore to separate the processing date
            scene_id = "_".join(item.id.split("_")[:-1]) 
            grouped_items[scene_id].append(item)

        # Find the best item (most recently processed) from each group
        best_items = []
        for scene_id, duplicate_items in grouped_items.items():
            if len(duplicate_items) > 1:
                # If there are duplicates, sort them by the full item ID (which includes the processing date)
                # and pick the last one (which will be the most recent)
                latest_item = sorted(duplicate_items, key=lambda x: x.id)[-1]
                best_items.append(latest_item)
            else:
                # If there's only one, it's the best by default
                best_items.append(duplicate_items[0])
                
        # --- MODIFICATION END ---

        stac_count = len(best_items)
        logger.info(f"Found {stac_count} unique images after filtering duplicates")
        
        return best_items
    
    def create_data_stack(self, items: list, bbox: box) -> xr.Dataset:
        """
        Create xarray data stack from signed STAC items.

        
        Args:
            items (list): List of STAC items
            bbox (box): Bounding box for data extraction
            
        Returns:
            xr.Dataset: Processed satellite data stack
        """
        logger.info("Creating data stack from STAC items...")

        # Get CRS
        crs_string = items[0].properties['proj:code']
        
        # Create xarray stack
        # epsg_text = self.satellite_config['crs']
        # epsg_code = int(epsg_text.split(':')[1])
        epsg_text = crs_string
        epsg_code = int(epsg_text.split(':')[1])
        
        stack = stackstac.stack(
            items,
            epsg=epsg_code,
            resolution=self.satellite_config['resolution'],
            assets=self.satellite_config['assets']
        )
        
        # Reproject AOI to satellite CRS
        # Bbox comes from WGS84 (EPSG:4326) - set CRS explicitly
        # The bbox from get_ds_sampled_mpc is always in WGS84 (EPSG:4326)
        aoi_gdf = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4326')
        aoi_reprojected = aoi_gdf.to_crs(crs_string)
        xmin, ymin, xmax, ymax = aoi_reprojected.total_bounds
        
        # Select subset
        subset_stack = stack.sel(x=slice(xmin, xmax), y=slice(ymax, ymin)) # For STAC APPROACH this ymax, ymin is required, but for GEE, it is ymin,ymax
        
        # Drop unnecessary variables
        subset_stack = self._drop_unnecessary_vars(subset_stack)
        
        # Convert bands to data variables
        subset_stack = subset_stack.to_dataset(dim="band")
        
        return subset_stack
    
    def _drop_unnecessary_vars(self, stack: xr.Dataset) -> xr.Dataset:
        """
        Drop unnecessary variables from the data stack.
        
        Args:
            stack (xr.Dataset): Input data stack
            
        Returns:
            xr.Dataset: Stack with unnecessary variables removed
        """
        drop_vars = [
            's2:high_proba_clouds_percentage', 'sat:orbit_state', 's2:unclassified_percentage',
            's2:snow_ice_percentage', 's2:datastrip_id', 's2:mgrs_tile',
            's2:reflectance_conversion_factor', 's2:degraded_msi_data_percentage',
            's2:saturated_defective_pixel_percentage', 's2:datatake_type',
            's2:mean_solar_azimuth', 'instruments', 's2:thin_cirrus_percentage',
            's2:dark_features_percentage', 's2:granule_id', 'eo:cloud_cover',
            's2:vegetation_percentage', 'sat:relative_orbit', 's2:mean_solar_zenith',
            's2:cloud_shadow_percentage', 's2:nodata_pixel_percentage',
            's2:product_type', 's2:generation_time', 's2:product_uri',
            'constellation', 's2:processing_baseline', 's2:water_percentage',
            's2:datatake_id', 'proj:bbox', 's2:not_vegetated_percentage',
            's2:medium_proba_clouds_percentage', "full_width_half_max", 
            "center_wavelength", "common_name", "title", "gsd"
        ]
        
        for var in drop_vars:
            if var in stack.coords:
                stack = stack.drop_vars([var])
        
        return stack
    
    def apply_cloud_masking(self, data: xr.Dataset) -> xr.Dataset:
        """
        Apply cloud masking using SCL (Scene Classification Layer).
        
        Args:
            data (xr.Dataset): Input dataset with SCL band
            
        Returns:
            xr.Dataset: Dataset with cloud mask applied
        """
        logger.info("Applying cloud masking...")
        
        # Calculate cloud mask from SCL band
        scl = data['SCL']
        is_cloud = (scl == 0) + (scl == 1) + (scl == 3) + \
                  (scl == 8) + (scl == 9) + (scl == 10) + np.isnan(scl)
        
        # Add validity mask
        data['is_valid'] = ~(is_cloud.astype(bool))
        
        return data
    
    def rename_bands(self, data: xr.Dataset) -> xr.Dataset:
        """
        Rename bands according to configuration mapping.
        
        Args:
            data (xr.Dataset): Input dataset with original band names
            
        Returns:
            xr.Dataset: Dataset with renamed bands
        """
        band_mapping = self.satellite_config.get('band_mapping', {})
        
        if not band_mapping:
            logger.warning("No band mapping found in config. Using original band names.")
            return data
        
        rename_dict = {}
        
        # Create rename dictionary for bands that exist in the data
        for old_name, new_name in band_mapping.items():
            if old_name in data.data_vars:
                rename_dict[old_name] = new_name
                logger.debug(f"Renaming band {old_name} -> {new_name}")
        
        if rename_dict:
            data = data.rename(rename_dict)
            logger.info(f"Renamed {len(rename_dict)} bands according to configuration")
        else:
            logger.warning("No bands found to rename according to mapping")
        
        return data
    
    
    def save_to_zarr(self, data: xr.Dataset, out_path: str, gcp_bucket=None) -> None:
        """
        Save dataset to Zarr format.
        
        Args:
            data (xr.Dataset): Dataset to save
            out_path (str): Output path for Zarr storage
        """
        
#         # Convert attributes to strings for Zarr compatibility
#         data.attrs['transform'] = str(data.attrs['transform'])
#         data.attrs['spec'] = str(data.attrs['spec'])

#         # Save with chunking
#         data.chunk({'time': 1000, 'x': 128, 'y': 128})

        with ProgressBar():
            if gcp_bucket is None:
                logger.info(f"Saving data to {out_path}")
                # Ensure output directory exists
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                data.to_zarr(out_path, mode="w", compute=True)
            else:
                logger.info(f"Saving data to {gcp_bucket}")
                # The gcsfs library handles authentication automatically in Vertex AI
                gcs = gcsfs.GCSFileSystem()
                # Create a mapper object that xarray can write to
                mapper = gcs.get_mapper(gcp_bucket)
                data.to_zarr(mapper, mode="w", consolidated=True, compute=True)
    
    def load_from_zarr(self, zarr_path: str) -> xr.Dataset:
        """
        Load dataset from Zarr format.
        
        Args:
            zarr_path (str): Path to Zarr storage
            
        Returns:
            xr.Dataset: Loaded dataset
        """
        if os.path.exists(zarr_path):
            logger.info(f"Loading existing data from {zarr_path}")
            return xr.open_zarr(zarr_path)
        else:
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
    
    def process_satellite_data(self, bbox: box, out_path: Optional[str] = None, 
                              show_progress: bool = True, gcp_bucket=None,
                              datetime_range: Optional[str] = None) -> xr.Dataset:
        """
        Simple STAC-based satellite data processing pipeline.
        
        Args:
            bbox (box): Bounding box for data extraction
            out_path (Optional[str]): Output path for Zarr storage
            show_progress (bool): Whether to show progress bars
            gcp_bucket: GCP bucket for storage (deprecated, use out_path with gs://)
            datetime_range (Optional[str]): Date range for search (e.g., "2017-01-01/2024-12-31").
                If None, uses datetime from config.
            
        Returns:
            xr.Dataset: Processed satellite data
        """
        # Check if data already exists
        if out_path and os.path.exists(out_path):
            return self.load_from_zarr(out_path)
        
        if gcp_bucket != None:
            import gcsfs
        
        # Start timing
        start_time = time.time()
        logger.info("🚀 Starting simple STAC satellite data processing pipeline...")
        print("🚀 Starting simple STAC satellite data processing pipeline...")
        
        # Calculate AOI area for time estimation
        aoi_gdf = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4326')
        # Use output_crs from config if available, otherwise use a default UTM (EPSG:32636)
        output_crs = self.config.get('output_crs') or 'EPSG:32636'
        aoi_utm = aoi_gdf.to_crs(output_crs)
        area_ha = aoi_utm.geometry.area.sum() / 10000
        
        print(f"📊 AOI Area: {area_ha:.1f} hectares")
        print("=" * 60)
        
        try:
            # Step 1: Search for satellite data
            step_start = time.time()
            print("🔍 Step 1/5: Searching for satellite data...")
            items = self.search_satellite_data(bbox, datetime_range=datetime_range)
            step_time = time.time() - step_start
            print(f"✅ Found {len(items)} images in {step_time:.1f}s")
            
            # Step 2: Create data stack
            step_start = time.time()
            print("📦 Step 2/5: Creating data stack...")
            data_stack = self.create_data_stack(items, bbox)
            step_time = time.time() - step_start
            print(f"✅ Data stack created in {step_time:.1f}s")
            
            # Step 3: Apply cloud masking
            step_start = time.time()
            print("☁️  Step 3/5: Applying cloud masking...")
            data_stack = self.apply_cloud_masking(data_stack)
            step_time = time.time() - step_start
            print(f"✅ Cloud masking applied in {step_time:.1f}s")
            
            # Step 4: Rename bands
            step_start = time.time()
            print("🏷️  Step 4/5: Renaming bands...")
            data_stack = self.rename_bands(data_stack)
            step_time = time.time() - step_start
            print(f"✅ Bands renamed in {step_time:.1f}s")
        
            # Step 5: Compute data (this is the longest step)
            step_start = time.time()
            print("💾 Step 5/5: Computing and downloading data...")
            print("   📡 Downloading satellite imagery...")
            
            ## avoid computed data, just use zarr later, because big AOI requires big memory
            # # num_workers = os.cpu_count() * 3
            # # num_workers = 6
            # if show_progress:
            #     with dask.diagnostics.ProgressBar():
            # #         computed_data = data_stack.compute(
            # #             scheduler="processes", num_workers=num_workers)
            #         # Threads are much more memory-efficient
            #         computed_data = data_stack.compute(scheduler="threads", num_workers=8) 
            # else:
            #     # computed_data = data_stack.compute(
            # #         scheduler="processes", num_workers=num_workers)
            #     # Threads are much more memory-efficient
            #     # computed_data = data_stack.compute(scheduler="threads", num_workers=8) 
            
            # self.data = computed_data
            # print('computed data is done!')
                  
            # step_time = time.time() - step_start
            # print(f"✅ Data computed and downloaded in {step_time:.1f}s")
                  
            # we save directly to zarr first (PUT IN THE CELL NOTEBOOK CELL LATER), later just open zarr, because compute will require to download entire dataset (not lazy loading)           
            # Step 6: Save to Zarr if path provided
#             if out_path:
#                 step_start = time.time()
                
#                 print("💾 Saving to Zarr storage...")
#                 self.save_to_zarr(computed_data, out_path, gcp_bucket = gcp_bucket)
                
#                 # Use dask.diagnostics to see the progress
#                 # with dask.diagnostics.ProgressBar():
#                 #     # The to_zarr method computes and saves chunk by chunk, avoiding high memory usage
#                 #     data_stack.to_zarr(zarr_path, mode='w', compute=True, consolidated=True)
                
#                 step_time = time.time() - step_start
#                 print(f"✅ Data saved to Zarr in {step_time:.1f}s")
            
            # Final timing summary
            total_time = time.time() - start_time
            total_minutes = total_time / 60
            
            print("=" * 60)
            print(f"🎉 STAC Processing Complete, please proceed to download to local first, and also to the gcp_bucket!")
            print(f"⏱️  Total time: {total_minutes:.1f} minutes ({total_time:.1f}s)")
            
            # return computed_data
            ###return the data stack instead, because kernel died issue
            return data_stack
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"❌ Processing failed after {total_time:.1f}s")
            raise
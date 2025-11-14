"""
STAC Utility Functions for MPC Eligibility Analysis

This module provides utilities for analyzing STAC (SpatioTemporal Asset Catalog) data,
specifically for finding duplicate scenes with different processing versions from
Microsoft Planetary Computer (MPC).
"""

import time
import logging
from collections import defaultdict
from typing import Optional, Dict, Any, List, Tuple
from shapely.geometry import box
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_stac_id(stac_id: str) -> Tuple[Optional[str], ...]:
    """
    Parse STAC ID to extract scene components.
    
    Format: S2A_MSIL2A_20240607T075611_R035_T36NVH_20240607T134827
    Returns: (satellite, product, acquisition_time, orbit, tile, processing_time)
    
    Args:
        stac_id: STAC item ID string
        
    Returns:
        Tuple of parsed components or (None, None, None, None, None, None) if parsing fails
    """
    parts = stac_id.split('_')
    if len(parts) >= 6:
        satellite = parts[0]  # S2A, S2B, S2C
        product = parts[1]    # MSIL2A
        acquisition_time = parts[2]  # 20240607T075611
        orbit = parts[3]      # R035
        tile = parts[4]       # T36NVH
        processing_time = parts[5]   # 20240607T134827
        return satellite, product, acquisition_time, orbit, tile, processing_time
    return None, None, None, None, None, None


def find_duplicate_stac_scenes(stac_data: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Find scenes with multiple processing versions from STAC data.
    
    Args:
        stac_data: List of STAC items
        
    Returns:
        dict: Dictionary of duplicate scenes with their versions
        Format: {scene_key: [version_dict, ...]}
    """
    # Group images by scene (same satellite, acquisition time, orbit, tile)
    scene_groups = defaultdict(list)

    for item in stac_data:
        stac_id = item.id
        parsed = parse_stac_id(stac_id)
        if parsed[0]:  # If parsing successful
            satellite, product, acquisition_time, orbit, tile, processing_time = parsed
            # Create scene key (everything except processing time)
            scene_key = f"{satellite}_{product}_{acquisition_time}_{orbit}_{tile}"
            scene_groups[scene_key].append({
                'item': item,
                'full_id': stac_id,
                'processing_time': processing_time,
                'cloud_cover': item.properties.get('eo:cloud_cover', None),
                'datetime': item.properties.get('datetime', ''),
                'geometry': item.geometry,
                'bbox': item.bbox
            })

    # Find scenes with multiple processing versions
    duplicate_scenes = {k: v for k, v in scene_groups.items() if len(v) > 1}
    
    return duplicate_scenes


def create_fresh_stac_client(config: Dict[str, Any]):
    """
    Create a completely fresh STAC client with new authentication.
    
    Args:
        config: Configuration dictionary with 'url_satellite_cloud' key
        
    Returns:
        pystac_client.Client: Fresh STAC client
    """
    import pystac_client
    import planetary_computer
    
    logger.info("Creating fresh STAC client...")
    
    # Add delay to ensure fresh authentication
    time.sleep(2)
    
    # Create new client with fresh modifier
    catalog = pystac_client.Client.open(
        config.get('url_satellite_cloud', 'https://planetarycomputer.microsoft.com/api/stac/v1'),
        modifier=planetary_computer.sign_inplace,
    )
    
    return catalog


def search_and_analyze_duplicate_stac_scenes(
    config: Dict[str, Any],
    bbox: box,
    max_examples: int = 5,
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """
    Search for STAC images and find duplicate scenes for analysis with token refresh mechanism.
    
    Args:
        config: Configuration dictionary with STAC parameters:
            - 'collection_mpc': STAC collection name (e.g., 'sentinel-2-l2a')
            - 'date_range_mpc': Date range as [start, end] strings
            - 'satellite.cloud_cover_threshold': Cloud cover threshold percentage
            - 'url_satellite_cloud': STAC catalog URL
        bbox: Bounding box for search (shapely.geometry.box)
        max_examples: Maximum number of duplicate examples to show
        max_retries: Maximum number of retry attempts
        
    Returns:
        dict: Selected scene data for visualization with keys:
            - 'scene_key': Scene identifier
            - 'versions': List of version dictionaries
            - 'example_id': Example STAC item ID
        Returns None if no duplicates found, or dict with 'fallback_used': True if search failed
    """
    import pystac_client
    import planetary_computer
    from pystac_client.exceptions import APIError
    
    logger.info("FRESH STAC SEARCH - Find Duplicate Scenes with Different Suffixes")
    logger.info("=" * 70)
    
    collection_mpc = config.get('collection_mpc', 'sentinel-2-l2a')
    date_range_mpc = config.get('date_range_mpc')
    cloud_cover_threshold = config.get('satellite', {}).get('cloud_cover_threshold', 80)
    url_satellite_cloud = config.get('url_satellite_cloud', 'https://planetarycomputer.microsoft.com/api/stac/v1')
    
    logger.info(f"Configuration:")
    logger.info(f"   • Collection: {collection_mpc}")
    logger.info(f"   • Date range: {date_range_mpc}")
    logger.info(f"   • Cloud cover threshold: {cloud_cover_threshold}%")
    logger.info(f"   • Bounding box: {bbox.bounds}")
    
    def create_catalog_with_retry():
        """Create STAC catalog with retry mechanism"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to STAC catalog (attempt {attempt + 1}/{max_retries})...")
                
                # Create fresh client on retry attempts
                if attempt > 0:
                    logger.info("Creating fresh STAC client...")
                    catalog = create_fresh_stac_client(config)
                else:
                    # First attempt - use standard approach
                    catalog = pystac_client.Client.open(
                        url_satellite_cloud,
                        modifier=planetary_computer.sign_inplace,
                    )
                
                # Test the connection with a simple request
                logger.info("Testing connection...")
                catalog.get_collection(collection_mpc)
                logger.info("Connection successful!")
                
                return catalog
                
            except APIError as e:
                logger.warning(f"API Error (attempt {attempt + 1}): {e}")
                if "maximum allowed time" in str(e) or "timeout" in str(e).lower():
                    logger.info("Request timeout - likely token expiration")
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                elif "401" in str(e) or "unauthorized" in str(e).lower():
                    logger.info("Authentication error - refreshing token")
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                else:
                    logger.error(f"Unrecoverable API error: {e}")
                    break
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    break
        
        raise Exception(f"Failed to connect to STAC catalog after {max_retries} attempts")
    
    def perform_search_with_retry(catalog):
        """Perform STAC search with retry mechanism"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Searching for Sentinel-2 images (attempt {attempt + 1}/{max_retries})...")
                
                # Use original date range - no reduction
                datetime_param = date_range_mpc
                
                # Use smaller limit and pagination to avoid timeouts
                page_size = 100 if attempt == 0 else 50  # Smaller pages on retry
                
                search = catalog.search(
                    collections=[collection_mpc],
                    intersects=bbox,
                    datetime=datetime_param,
                    query={"eo:cloud_cover": {"lt": cloud_cover_threshold}},
                    limit=page_size
                )
                
                # Get items with timeout handling
                logger.info(f"   Fetching results (page size: {page_size})...")
                items_stac = search.item_collection()
                logger.info(f"Found {len(items_stac)} images")
                return items_stac
                
            except APIError as e:
                logger.warning(f"Search API Error (attempt {attempt + 1}): {e}")
                if "maximum allowed time" in str(e) or "timeout" in str(e).lower():
                    logger.info("Search timeout - retrying with fresh connection")
                    if attempt < max_retries - 1:
                        # Create fresh catalog for retry
                        catalog = create_fresh_stac_client(config)
                        wait_time = (attempt + 1) * 5  # Longer wait for timeouts
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                else:
                    logger.error(f"Unrecoverable search error: {e}")
                    break
            except Exception as e:
                logger.error(f"Unexpected search error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    break
        
        raise Exception(f"Failed to perform STAC search after {max_retries} attempts")
    
    # Create catalog with retry mechanism
    try:
        catalog = create_catalog_with_retry()
        
        # Perform search with retry mechanism
        items_stac = perform_search_with_retry(catalog)
        
    except Exception as e:
        logger.error(f"STAC search completely failed: {e}")
        logger.info("Attempting fallback approach...")
        
        # Fallback: Return empty results with warning
        logger.warning("Using fallback: Returning empty STAC results")
        logger.info("Consider:")
        logger.info("   • Reducing the date range in config.json")
        logger.info("   • Using a smaller AOI")
        logger.info("   • Increasing cloud cover threshold")
        logger.info("   • Using GEE approach instead (use_gee=True)")
        
        # Return empty structure that won't break the rest of the code
        return {
            'selected_scenes': [],
            'duplicate_groups': {},
            'total_scenes': 0,
            'duplicate_count': 0,
            'fallback_used': True,
            'error_message': str(e)
        }
    
    # Find duplicate scenes
    duplicate_scenes = find_duplicate_stac_scenes(items_stac)
    
    logger.info(f"Analysis Results:")
    logger.info(f"   • Total images: {len(items_stac)}")
    unique_scenes = len(duplicate_scenes) + (len(items_stac) - sum(len(v) for v in duplicate_scenes.values()))
    logger.info(f"   • Unique scenes: {unique_scenes}")
    logger.info(f"   • Scenes with multiple processing versions: {len(duplicate_scenes)}")
    
    if duplicate_scenes:
        logger.info(f"SCENES WITH MULTIPLE PROCESSING VERSIONS:")
        logger.info("=" * 60)
        
        # Show first few examples
        for i, (scene_key, versions) in enumerate(list(duplicate_scenes.items())[:max_examples]):
            logger.info(f"{i+1}. Scene: {scene_key}")
            logger.info(f"   Versions: {len(versions)}")
            
            for j, version in enumerate(versions):
                logger.info(f"   {j+1}: {version['full_id']}")
                logger.info(f"      Processing: {version['processing_time']}")
                logger.info(f"      Cloud Cover: {version['cloud_cover']:.2f}%")
                logger.info(f"      DateTime: {version['datetime']}")
            
            # Check if cloud cover differs significantly
            cloud_covers = [v['cloud_cover'] for v in versions if v['cloud_cover'] is not None]
            if len(cloud_covers) > 1:
                cloud_diff = max(cloud_covers) - min(cloud_covers)
                logger.info(f"   ⚠️  Cloud cover difference: {cloud_diff:.2f}%")
        
        # Select the first example for detailed investigation
        first_duplicate = list(duplicate_scenes.items())[0]
        selected_scene_key, selected_versions = first_duplicate
        
        logger.info(f"SELECTED EXAMPLE FOR INVESTIGATION:")
        logger.info("=" * 50)
        logger.info(f"Scene: {selected_scene_key}")
        logger.info(f"Versions: {len(selected_versions)}")
        
        for i, version in enumerate(selected_versions):
            logger.info(f"   {i+1}: {version['full_id']}")
        
        # Store the selected example data
        selected_scene_data = {
            'scene_key': selected_scene_key,
            'versions': selected_versions,
            'example_id': selected_versions[0]['full_id']
        }
        
        logger.info(f"Selected example ID: {selected_scene_data['example_id']}")
        logger.info(f"   This will be used for detailed comparison")
        
        return selected_scene_data
        
    else:
        logger.info("No duplicate scenes found in the current search")
        logger.info("   All scenes appear to have unique processing versions")
        return None


def create_unified_stac_visualization(
    gdf: Any,
    bbox: box,
    scene_data: Optional[Dict[str, Any]] = None,
    save_map: bool = True,
    map_filename: Optional[str] = None
) -> Any:
    """
    Create a unified map visualization that combines:
    - Area of interest (AOI)
    - Bounding box
    - STAC scene versions (if provided)
    
    Args:
        gdf: GeoDataFrame with area of interest
        bbox: Bounding box for the search area
        scene_data: Optional STAC scene data for visualization
        save_map: Whether to save the map as HTML file
        map_filename: Custom filename for the saved map
        
    Returns:
        folium.Map: Interactive map object with all layers
    """
    import folium
    import json
    
    # Use both logger and print for notebook visibility
    logger.info("🗺️ CREATING UNIFIED STAC VISUALIZATION")
    print("🗺️ CREATING UNIFIED STAC VISUALIZATION")
    logger.info("=" * 50)
    print("=" * 50)
    
    # Create base map
    center_lat = (bbox.bounds[1] + bbox.bounds[3]) / 2
    center_lon = (bbox.bounds[0] + bbox.bounds[2]) / 2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Create layer groups for better organization
    aoi_group = folium.FeatureGroup(name="Area of Interest (AOI)")
    bbox_group = folium.FeatureGroup(name="Search Bounding Box")
    
    # Add the area of interest to its layer group
    if gdf is not None:
        geojson_data = json.loads(gdf.to_json())
        folium.GeoJson(
            geojson_data,
            style_function=lambda feature: {
                'fillColor': 'red',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0.3
            },
            popup=folium.Popup("Area of Interest", parse_html=True)
        ).add_to(aoi_group)
    
    # Add bounding box to its layer group
    bbox_coords = [
        [bbox.bounds[1], bbox.bounds[0]],  # SW
        [bbox.bounds[3], bbox.bounds[0]],  # NW
        [bbox.bounds[3], bbox.bounds[2]],  # NE
        [bbox.bounds[1], bbox.bounds[2]],  # SE
        [bbox.bounds[1], bbox.bounds[0]]   # Close the polygon
    ]
    
    folium.Polygon(
        bbox_coords,
        color='blue',
        weight=2,
        fillColor='blue',
        fillOpacity=0.1,
        popup="Bounding Box"
    ).add_to(bbox_group)
    
    # Add base layer groups to map
    aoi_group.add_to(m)
    bbox_group.add_to(m)
    
    # Add STAC scene versions if provided
    if scene_data and 'versions' in scene_data:
        versions = scene_data['versions']
        scene_key = scene_data['scene_key']
        
        logger.info(f"Adding STAC scene versions: {scene_key}")
        print(f"Adding STAC scene versions: {scene_key}")
        logger.info(f"Found {len(versions)} versions to overlay")
        print(f"Found {len(versions)} versions to overlay")
        
        # Define colors for different versions
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'darkred', 'lightblue']
        
        # Add each version as a separate layer group
        for i, version in enumerate(versions):
            color = colors[i % len(colors)]
            
            # Create a layer group for this version with a unique name
            layer_name = f"STAC Version {i+1}: {version['processing_time']}"
            layer_group = folium.FeatureGroup(name=layer_name)
            
            # Get the geometry from the STAC item
            geometry = version['geometry']
            
            # Create popup text with version details
            popup_text = f"""
            <b>STAC Version {i+1}</b><br>
            <b>ID:</b> {version['full_id']}<br>
            <b>Processing:</b> {version['processing_time']}<br>
            <b>Cloud Cover:</b> {version['cloud_cover']:.2f}%<br>
            <b>DateTime:</b> {version['datetime']}<br>
            <b>BBox:</b> [{version['bbox'][0]:.6f}, {version['bbox'][1]:.6f}, {version['bbox'][2]:.6f}, {version['bbox'][3]:.6f}]
            """
            
            # Add polygon to the layer group
            folium.GeoJson(
                geometry,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 3,
                    'fillOpacity': 0.4,
                    'opacity': 0.8
                },
                popup=folium.Popup(popup_text, max_width=400),
                tooltip=f"STAC Version {i+1}: {version['processing_time']}"
            ).add_to(layer_group)
            
            # Add the layer group to the map
            layer_group.add_to(m)
            
            logger.info(f"   {i+1}. {version['full_id']} - {color} boundary")
            print(f"   {i+1}. {version['full_id']} - {color} boundary")
        
        # Add a comparison layer showing all STAC versions together
        stac_comparison_group = folium.FeatureGroup(name="All STAC Versions (Comparison)", show=False)
        
        for i, version in enumerate(versions):
            color = colors[i % len(colors)]
            geometry = version['geometry']
            
            # Add with different opacity for comparison
            folium.GeoJson(
                geometry,
                style_function=lambda x, color=color: {
                    'fillColor': color,
                    'color': color,
                    'weight': 2,
                    'fillOpacity': 0.2,
                    'opacity': 0.6
                },
                popup=folium.Popup(f"STAC Version {i+1}: {version['processing_time']}", max_width=200),
                tooltip=f"STAC Version {i+1}: {version['processing_time']}"
            ).add_to(stac_comparison_group)
        
        stac_comparison_group.add_to(m)
        
        logger.info(f"✅ Added {len(versions)} STAC scene version layers")
        print(f"✅ Added {len(versions)} STAC scene version layers")
    else:
        logger.info("ℹ️ No STAC scene data provided - showing only base layers")
        print("ℹ️ No STAC scene data provided - showing only base layers")
    
    # Add layer control to the map
    layer_control = folium.LayerControl()
    layer_control.add_to(m)
    
    # Add fullscreen toggle button
    try:
        from folium.plugins import Fullscreen
        fullscreen = Fullscreen(
            position='topright',
            title='Toggle Fullscreen',
            title_cancel='Exit Fullscreen',
            force_separate_button=True
        )
        fullscreen.add_to(m)
    except ImportError:
        logger.warning("⚠️ folium.plugins.Fullscreen not available, skipping fullscreen button")
    
    # Add a comprehensive legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 300px; height: 200px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>🗺️ UNIFIED STAC VISUALIZATION</b></p>
    <p><i class="fa fa-square" style="color:red"></i> Area of Interest (AOI)</p>
    <p><i class="fa fa-square" style="color:blue"></i> Search Bounding Box</p>
    '''
    
    if scene_data and 'versions' in scene_data:
        legend_html += '<p><b>STAC Scene Versions:</b></p>'
        for i, version in enumerate(versions):
            color = colors[i % len(colors)]
            legend_html += f'<p><i class="fa fa-square" style="color:{color}"></i> Version {i+1}: {version["processing_time"]}</p>'
    
    legend_html += '''
    <p><b>🎛️ Instructions:</b></p>
    <p>• Use layer control (top-right) to toggle layers</p>
    <p>• Use fullscreen button (top-right) for fullscreen view</p>
    <p>• "All STAC Versions" shows overlap comparison</p>
    <p>• Click on features for detailed information</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map if requested
    if save_map:
        if not map_filename:
            if scene_data:
                map_filename = f"unified_stac_visualization_{scene_data['scene_key'].replace('_', '-')}.html"
            else:
                map_filename = "unified_exploration_map.html"
        m.save(map_filename)
        logger.info(f"\n✅ Unified map saved as: {map_filename}")
        logger.info(f"🗺️ Open this file in your browser to explore all layers")
        print(f"\n✅ Unified map saved as: {map_filename}")
        print(f"🗺️ Open this file in your browser to explore all layers")
    
    # Debug: Verify map is being returned
    logger.info(f"Returning folium map object: {type(m)}")
    print(f"Returning folium map object: {type(m)}")
    if m is None:
        logger.error("❌ Map object is None!")
        print("❌ Map object is None!")
    
    return m


def debug_stac_scene_data(scene_data: Optional[Dict[str, Any]]) -> bool:
    """
    Debug function to check STAC scene data structure.
    
    Args:
        scene_data: STAC scene data to debug
        
    Returns:
        bool: True if scene_data is valid, False otherwise
    """
    logger.info("DEBUGGING STAC SCENE DATA:")
    logger.info("=" * 40)
    
    if not scene_data:
        logger.warning("No scene_data provided")
        return False
    
    logger.info(f"Scene data type: {type(scene_data)}")
    logger.info(f"Scene data keys: {list(scene_data.keys()) if isinstance(scene_data, dict) else 'Not a dict'}")
    
    if 'versions' in scene_data:
        versions = scene_data['versions']
        logger.info(f"Number of versions: {len(versions)}")
        
        for i, version in enumerate(versions):
            logger.info(f"Version {i+1}:")
            logger.info(f"   • Keys: {list(version.keys()) if isinstance(version, dict) else 'Not a dict'}")
            logger.info(f"   • Full ID: {version.get('full_id', 'Missing')}")
            logger.info(f"   • Processing time: {version.get('processing_time', 'Missing')}")
            logger.info(f"   • Geometry type: {type(version.get('geometry', 'Missing'))}")
            logger.info(f"   • Bbox: {version.get('bbox', 'Missing')}")
        
        return True
    else:
        logger.warning("No 'versions' key found in scene_data")
        return False


def calculate_mpc_date_range(
    ds_resampled: 'xr.Dataset',
    years_back: int = 11,
    days_offset: int = 15
) -> Tuple[str, str]:
    """
    Calculate MPC date range based on ds_resampled dates.
    
    Simple calculation:
    - Start date: max_date - years_back years
    - End date: min_date - days_offset days
    
    Args:
        ds_resampled: xarray Dataset with time coordinate
        years_back: Number of years to go back from max_date for start date. Default 11.
        days_offset: Days to subtract from min_date for end date. Default 15.
        
    Returns:
        Tuple of (start_date, end_date) as strings in 'YYYY-MM-DD' format
    """
    import pandas as pd
    import numpy as np
    
    # Get max and min dates from dataset
    max_date = ds_resampled.time.max().values
    min_date = ds_resampled.time.min().values
    
    # Calculate start date: max_date - years_back years
    start_date_pd = pd.Timestamp(max_date) - pd.DateOffset(years=years_back)
    start_date_np = np.datetime64(start_date_pd, 'D')
    start_date = np.datetime_as_string(start_date_np, unit='D')
    
    # Calculate end date: min_date - days_offset days
    end_date_pd = pd.Timestamp(min_date) - pd.DateOffset(days=days_offset)
    end_date_np = np.datetime64(end_date_pd, 'D')
    end_date = np.datetime_as_string(end_date_np, unit='D')
    
    return start_date, end_date


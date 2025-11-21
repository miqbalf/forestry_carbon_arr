"""
Complete PlanetScope Workflow:
1. Load AOI shapefile from GCS
2. Search PlanetScope images with filters
3. Visualize results on interactive map with toggleable tile layers
4. Create order for download (with harmonization and COG support)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

import geopandas as gpd
import folium
from folium import plugins
import gcsfs

from .downloader import PlanetScopeDownloader


class PlanetScopeWorkflow:
    """
    Complete workflow for PlanetScope image search, visualization, and ordering.
    """
    
    def __init__(self, api_key: Optional[str] = None, gcs_project: Optional[str] = None):
        """
        Initialize workflow with Planet API and GCS access.
        
        Args:
            api_key: Planet API key (or set PLANET_API_KEY env var)
            gcs_project: Google Cloud Project ID (or set GOOGLE_CLOUD_PROJECT env var)
        """
        # Use PLANET_API_KEY as primary reference
        self.api_key = api_key or os.getenv('PLANET_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set PLANET_API_KEY env var "
                "or pass api_key parameter."
            )
        
        # Initialize downloader (it will handle setting PL_API_KEY if needed)
        self.downloader = PlanetScopeDownloader(api_key=self.api_key)
        self.gcs_project = gcs_project or os.getenv('GOOGLE_CLOUD_PROJECT')
    
    def get_planet_tile_url(self, item_ids: List[str], item_type: str = 'PSScene') -> str:
        """
        Get Planet tiles URL for mosaic preview.
        Format: https://tiles{0-3}.planet.com/data/v1/{item_type}/{item_ids}/{z}/{x}/{y}.png?api_key={api_key}
        
        Args:
            item_ids: List of item IDs to create mosaic
            item_type: Item type (default: PSScene)
            
        Returns:
            Tile URL template
        """
        if not self.api_key:
            raise ValueError("API key required for tiles")
        
        # Join item IDs with commas
        items_str = ','.join(item_ids)
        
        # Use tiles2 (or tiles0-3 for load balancing)
        tile_base = f"https://tiles2.planet.com/data/v1/{item_type}/{items_str}/{{z}}/{{x}}/{{y}}.png?api_key={self.api_key}"
        
        return tile_base
    
    def group_items_by_date(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group items by acquisition date.
        
        Args:
            items: List of PlanetScope items
            
        Returns:
            Dictionary mapping date (YYYY-MM-DD) to list of items
        """
        grouped = defaultdict(list)
        
        for item in items:
            props = item.get('properties', {})
            acquired = props.get('acquired', '')
            
            if acquired:
                # Extract date part (YYYY-MM-DD)
                date_str = acquired[:10] if len(acquired) >= 10 else acquired.split('T')[0]
                grouped[date_str].append(item)
        
        return dict(grouped)
        
    def load_aoi_from_gcs(
        self, 
        gcs_path: str,
        to_crs: Optional[str] = 'EPSG:4326'  # WGS84 for Planet API
    ) -> gpd.GeoDataFrame:
        """
        Load shapefile from Google Cloud Storage.
        
        Args:
            gcs_path: GCS path to shapefile (e.g., 'gs://bucket/path/to/file.shp')
            to_crs: Target CRS (default WGS84 for Planet API)
            
        Returns:
            GeoDataFrame with AOI geometry
        """
        print(f"📂 Loading AOI from GCS: {gcs_path}")
        
        # Initialize GCS filesystem
        if not self.gcs_project:
            raise ValueError(
                "GCS project required. Set GOOGLE_CLOUD_PROJECT env var "
                "or pass gcs_project parameter."
            )
        
        # Setup GCS filesystem (similar to your codebase pattern)
        token_path = os.getenv("GCS_TOKEN_PATH", "/usr/src/app/user_id.json")
        fs_kwargs = {"project": self.gcs_project}
        if token_path and os.path.exists(token_path):
            fs_kwargs["token"] = token_path
        
        fs = gcsfs.GCSFileSystem(**fs_kwargs)
        
        if not fs.exists(gcs_path):
            raise FileNotFoundError(f"AOI file not found on GCS: {gcs_path}")
        
        # Read shapefile from GCS
        gdf = gpd.read_file(gcs_path, filesystem=fs)
        
        print(f"✅ Loaded AOI: {len(gdf)} features")
        print(f"   CRS: {gdf.crs}")
        print(f"   Bounds: {gdf.total_bounds}")
        
        # Convert to WGS84 if needed (Planet API requires WGS84)
        if to_crs and gdf.crs != to_crs:
            print(f"🔄 Converting CRS to {to_crs}...")
            gdf = gdf.to_crs(to_crs)
            print(f"   New bounds: {gdf.total_bounds}")
        
        return gdf
    
    def aoi_to_geojson(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Convert GeoDataFrame to GeoJSON for Planet API.
        Handles multiple geometries by creating a union or using the first feature.
        
        Args:
            gdf: GeoDataFrame with AOI
            
        Returns:
            GeoJSON geometry dict
        """
        # If multiple features, dissolve/union them
        if len(gdf) > 1:
            print(f"⚠️  Multiple features found ({len(gdf)}). Creating union...")
            gdf_union = gdf.unary_union
            # Convert back to GeoDataFrame
            gdf = gpd.GeoDataFrame(geometry=[gdf_union], crs=gdf.crs)
        
        # Get geometry from first (and only) feature
        geometry = gdf.geometry.iloc[0]
        
        # Convert to GeoJSON
        geojson = json.loads(gpd.GeoSeries([geometry]).to_json())
        geojson_geometry = geojson['features'][0]['geometry']
        
        print(f"✅ Converted AOI to GeoJSON")
        print(f"   Type: {geojson_geometry['type']}")
        
        return geojson_geometry
    
    def search_and_filter(
        self,
        aoi_geojson: Dict,
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 0.1,
        item_type: str = 'PSScene',
        limit: int = 100
    ) -> List[Dict]:
        """
        Search PlanetScope images with filters.
        
        Args:
            aoi_geojson: GeoJSON geometry
            start_date: Start date (ISO format: "2024-01-01T00:00:00Z")
            end_date: End date (ISO format: "2024-12-31T23:59:59Z")
            cloud_cover_max: Maximum cloud cover (0.0-1.0)
            item_type: Item type ('PSScene' or 'PSScene4Band')
            limit: Maximum results
            
        Returns:
            List of found items
        """
        print(f"\n🔍 Searching PlanetScope images...")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Max cloud cover: {cloud_cover_max*100}%")
        print(f"   Item type: {item_type}")
        
        items = self.downloader.search_images(
            geometry=aoi_geojson,
            start_date=start_date,
            end_date=end_date,
            item_type=item_type,
            cloud_cover_max=cloud_cover_max,
            limit=limit
        )
        
        return items
    
    def visualize_results(
        self,
        gdf_aoi: gpd.GeoDataFrame,
        items: List[Dict],
        add_tile_layers: bool = True
    ) -> folium.Map:
        """
        Create interactive map visualization of AOI and found images.
        Map is displayed in notebook, not saved to file.
        
        Args:
            gdf_aoi: GeoDataFrame with AOI
            items: List of PlanetScope items from search
            add_tile_layers: Whether to add toggleable Planet tile layers
            
        Returns:
            folium.Map object
        """
        print(f"\n🗺️  Creating visualization map...")
        
        # Calculate map center from AOI
        bounds = gdf_aoi.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
        
        # Create layer groups
        aoi_group = folium.FeatureGroup(name="Area of Interest (AOI)")
        scenes_group = folium.FeatureGroup(name=f"PlanetScope Scenes ({len(items)})")
        
        # Add AOI
        aoi_geojson = json.loads(gdf_aoi.to_json())
        folium.GeoJson(
            aoi_geojson,
            style_function=lambda feature: {
                'fillColor': 'red',
                'color': 'black',
                'weight': 3,
                'fillOpacity': 0.3,
                'dashArray': '5, 5'
            },
            popup="Area of Interest",
            tooltip="AOI"
        ).add_to(aoi_group)
        
        # Add PlanetScope scene footprints
        for i, item in enumerate(items):
            # Get geometry from item
            if 'geometry' in item:
                geometry = item['geometry']
            elif 'geometry' in item.get('properties', {}):
                geometry = item['properties']['geometry']
            else:
                continue
            
            # Get item properties
            props = item.get('properties', {})
            item_id = item.get('id', f'Item_{i}')
            acquired = props.get('acquired', 'Unknown')
            cloud_cover = props.get('cloud_cover', 0) * 100
            
            # Color based on cloud cover
            if cloud_cover < 5:
                color = 'green'
            elif cloud_cover < 10:
                color = 'orange'
            else:
                color = 'red'
            
            # Create simple popup text (no HTML)
            popup_text = f"PlanetScope Scene\nID: {item_id}\nDate: {acquired}\nCloud Cover: {cloud_cover:.1f}%\nType: {props.get('item_type', 'N/A')}"
            
            # Add footprint
            folium.GeoJson(
                geometry,
                style_function=lambda feature, c=color: {
                    'fillColor': c,
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                popup=popup_text,
                tooltip=f"{item_id[:20]}... ({cloud_cover:.1f}% clouds)"
            ).add_to(scenes_group)
        
        # Add layer groups to map
        aoi_group.add_to(m)
        scenes_group.add_to(m)
        
        # Add Planet tile layers grouped by date (mosaic previews)
        if add_tile_layers and items:
            try:
                # Group items by date
                items_by_date = self.group_items_by_date(items)
                
                print(f"   📅 Creating tile layers for {len(items_by_date)} dates...")
                
                # Add tile layer for each date
                for date_str, date_items in sorted(items_by_date.items()):
                    item_ids = [item.get('id') for item in date_items if item.get('id')]
                    
                    if not item_ids:
                        continue
                    
                    # Get tile URL for this date's mosaic
                    try:
                        tile_url = self.get_planet_tile_url(item_ids, item_type='PSScene')
                        
                        # Create layer name with date and count
                        layer_name = f"🌍 Planet Mosaic - {date_str} ({len(item_ids)} scenes)"
                        
                        # Add as toggleable tile layer
                        folium.TileLayer(
                            tiles=tile_url,
                            attr='Planet Labs',
                            name=layer_name,
                            overlay=True,
                            control=True,
                            show=False  # Hidden by default, can be toggled
                        ).add_to(m)
                        
                        print(f"      ✅ Added: {layer_name}")
                        
                    except Exception as e:
                        print(f"      ⚠️  Could not add tile layer for {date_str}: {e}")
                        continue
                
                print(f"   💡 Toggle date-based mosaics using the layer control (top-right)")
                
            except Exception as e:
                print(f"   ⚠️  Error adding tile layers: {e}")
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add fullscreen button
        plugins.Fullscreen().add_to(m)
        
        # Add measure tool
        plugins.MeasureControl().add_to(m)
        
        # Fit map to bounds
        if items:
            # Get all scene bounds
            all_bounds = [gdf_aoi.total_bounds]
            for item in items:
                if 'geometry' in item:
                    try:
                        scene_gdf = gpd.GeoDataFrame([1], geometry=[item['geometry']], crs='EPSG:4326')
                        all_bounds.append(scene_gdf.total_bounds)
                    except:
                        pass
            
            # Calculate combined bounds
            min_lon = min(b[0] for b in all_bounds)
            min_lat = min(b[1] for b in all_bounds)
            max_lon = max(b[2] for b in all_bounds)
            max_lat = max(b[3] for b in all_bounds)
            
            m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
        
        print(f"✅ Map created with {len(items)} scenes")
        print(f"   Map will be displayed in notebook (not saved to file)")
        
        return m
    
    def create_clipped_harmonized_order(
        self,
        item_ids: List[str],
        aoi_geojson: Dict,
        bundle_type: str = 'analytic_sr_udm2',
        order_name: Optional[str] = None
    ) -> Dict:
        """
        Create an order for harmonized 8-band PlanetScope mosaic clipped to AOI.
        
        Args:
            item_ids: List of item IDs to include in mosaic
            aoi_geojson: GeoJSON geometry for clipping
            bundle_type: Bundle type (default: 'analytic_sr_udm2' for 8-band harmonized)
            order_name: Optional order name
            
        Returns:
            Order details dictionary
        """
        return self.downloader.create_order_with_clip(
            item_ids=item_ids,
            aoi_geometry=aoi_geojson,
            bundle_type=bundle_type,
            order_name=order_name,
            harmonized=True
        )
    
    def download_and_convert_to_cog(
        self,
        order_id: str,
        output_dir: str = './downloads',
        convert_to_cog: bool = True,
        upload_to_gcs: Optional[str] = None
    ) -> List[str]:
        """
        Download order files and optionally convert to COG format.
        
        Args:
            order_id: Order ID
            output_dir: Output directory for downloads
            convert_to_cog: If True, convert downloaded files to COG
            upload_to_gcs: Optional GCS path to upload COG (e.g., 'gs://bucket/path/')
            
        Returns:
            List of downloaded/converted file paths
        """
        # Wait for order to complete
        order = self.downloader.wait_for_order(order_id)
        
        if order['state'] != 'success':
            print(f"❌ Order not successful. State: {order['state']}")
            return []
        
        # Download files
        downloaded_files = self.downloader.download_order(order_id, output_dir)
        
        if not convert_to_cog:
            return downloaded_files
        
        # Convert each file to COG
        cog_files = []
        for filepath in downloaded_files:
            if filepath.endswith('.tif') or filepath.endswith('.tiff'):
                cog_path = filepath.replace('.tif', '_COG.tif').replace('.tiff', '_COG.tif')
                cog_file = self.downloader.convert_to_cog(filepath, cog_path)
                if cog_file:
                    cog_files.append(cog_file)
                    
                    # Upload to GCS if specified
                    if upload_to_gcs:
                        self._upload_to_gcs(cog_file, upload_to_gcs)
        
        return cog_files if cog_files else downloaded_files
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str):
        """Upload file to Google Cloud Storage."""
        try:
            from google.cloud import storage
            import os
            
            # Parse GCS path
            if gcs_path.startswith('gs://'):
                gcs_path = gcs_path[5:]  # Remove 'gs://'
            
            parts = gcs_path.split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else os.path.basename(local_path)
            
            # Upload
            client = storage.Client(project=self.gcs_project)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            print(f"⬆️  Uploading to GCS: {gcs_path}")
            blob.upload_from_filename(local_path)
            print(f"✅ Uploaded: gs://{bucket_name}/{blob_name}")
            
        except Exception as e:
            print(f"⚠️  Failed to upload to GCS: {e}")
    
    def run_complete_workflow(
        self,
        gcs_shp_path: str,
        start_date: str,
        end_date: str,
        cloud_cover_max: float = 0.1,
        item_type: str = 'PSScene',
        limit: int = 100,
        visualize: bool = True,
        create_order: bool = False,
        selected_item_ids: Optional[List[str]] = None,
        create_clipped_order: bool = False,
        convert_to_cog: bool = False,
        upload_to_gcs: Optional[str] = None
    ) -> Dict:
        """
        Run complete workflow: load AOI, search, visualize, and optionally order.
        
        Args:
            gcs_shp_path: GCS path to shapefile
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            cloud_cover_max: Max cloud cover (0.0-1.0)
            item_type: Item type
            limit: Max search results
            visualize: Whether to create visualization map
            create_order: Whether to create download order
            selected_item_ids: Item IDs to include in order (if None, uses all found items)
            create_clipped_order: Whether to create clipped harmonized order
            convert_to_cog: Whether to convert downloaded files to COG
            upload_to_gcs: Optional GCS path to upload COG files
            
        Returns:
            Dictionary with results
        """
        results = {}
        
        # Step 1: Load AOI from GCS
        print("=" * 60)
        print("STEP 1: Loading AOI from GCS")
        print("=" * 60)
        gdf_aoi = self.load_aoi_from_gcs(gcs_shp_path)
        results['aoi'] = gdf_aoi
        
        # Step 2: Convert to GeoJSON
        print("\n" + "=" * 60)
        print("STEP 2: Converting AOI to GeoJSON")
        print("=" * 60)
        aoi_geojson = self.aoi_to_geojson(gdf_aoi)
        results['aoi_geojson'] = aoi_geojson
        
        # Step 3: Search images
        print("\n" + "=" * 60)
        print("STEP 3: Searching PlanetScope Images")
        print("=" * 60)
        items = self.search_and_filter(
            aoi_geojson=aoi_geojson,
            start_date=start_date,
            end_date=end_date,
            cloud_cover_max=cloud_cover_max,
            item_type=item_type,
            limit=limit
        )
        results['items'] = items
        results['item_count'] = len(items)
        
        # Step 4: Visualize
        if visualize:
            print("\n" + "=" * 60)
            print("STEP 4: Creating Visualization")
            print("=" * 60)
            map_obj = self.visualize_results(gdf_aoi, items, add_tile_layers=True)
            results['map'] = map_obj
            print("\n💡 Use the layer control (top-right) to toggle date-based mosaic previews!")
        
        # Step 5: Create order (optional)
        if create_order:
            print("\n" + "=" * 60)
            print("STEP 5: Creating Download Order")
            print("=" * 60)
            
            item_ids = selected_item_ids or [item['id'] for item in items]
            
            if not item_ids:
                print("⚠️  No items to order")
            else:
                order = self.downloader.create_order(
                    item_ids=item_ids,
                    bundle_type='analytic_sr_udm2'
                )
                results['order'] = order
                results['order_id'] = order.get('id')
        
        # Create clipped harmonized order
        if create_clipped_order:
            print("\n" + "=" * 60)
            print("STEP 6: Creating Clipped Harmonized Order (8-band)")
            print("=" * 60)
            
            item_ids = selected_item_ids or [item['id'] for item in items]
            
            if not item_ids:
                print("⚠️  No items to order")
            else:
                order = self.create_clipped_harmonized_order(
                    item_ids=item_ids,
                    aoi_geojson=aoi_geojson,
                    bundle_type='analytic_sr_udm2',  # 8-band harmonized
                    order_name=f"korindo_harmonized_clipped_{datetime.now().strftime('%Y%m%d')}"
                )
                results['clipped_order'] = order
                results['clipped_order_id'] = order.get('id')
                
                # Optionally download and convert to COG
                if convert_to_cog:
                    print("\n" + "=" * 60)
                    print("STEP 7: Downloading and Converting to COG")
                    print("=" * 60)
                    cog_files = self.download_and_convert_to_cog(
                        order_id=order['id'],
                        output_dir='./downloads',
                        convert_to_cog=True,
                        upload_to_gcs=upload_to_gcs
                    )
                    results['cog_files'] = cog_files
        
        print("\n" + "=" * 60)
        print("✅ WORKFLOW COMPLETE")
        print("=" * 60)
        
        return results


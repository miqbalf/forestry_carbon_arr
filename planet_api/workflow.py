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
from typing import Dict, List, Optional, Union
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
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        gcs_project: Optional[str] = None
    ):
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
        gcs_credentials: Optional[str] = None,
        to_crs: Optional[str] = 'EPSG:4326'  # WGS84 for Planet API
    ) -> gpd.GeoDataFrame:
        """
        Load shapefile from Google Cloud Storage.
        
        Args:
            gcs_path: GCS path to shapefile (e.g., 'gs://bucket/path/to/file.shp')
            gcs_credentials: Path to service account JSON file, JSON string, or base64-encoded JSON string
                           (optional, uses default credentials if not provided)
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
        
        # Setup GCS filesystem with credentials from argument
        fs_kwargs = {"project": self.gcs_project}
        
        if gcs_credentials:
            # If it's a file path, use it directly
            if os.path.exists(gcs_credentials):
                fs_kwargs["token"] = gcs_credentials
            else:
                # Otherwise, parse it (JSON string or base64) and create temp file
                creds = self._get_gcs_credentials(gcs_credentials)
                if creds:
                    import tempfile
                    import json
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                        json.dump(creds, tmp)
                        tmp_path = tmp.name
                    fs_kwargs["token"] = tmp_path
        
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
    
    def simplify_aoi_geometry(self, aoi_geojson: Dict, max_vertices: int = 1000) -> Dict:
        """
        Simplify AOI geometry to reduce vertex count for Planet API.
        Planet API has a limit of 1500 vertices.
        
        Args:
            aoi_geojson: GeoJSON geometry dictionary
            max_vertices: Maximum number of vertices (default: 1000, safe limit)
            
        Returns:
            Simplified GeoJSON geometry dictionary
        """
        try:
            from shapely.geometry import shape, mapping
            from shapely.ops import unary_union
            
            # Convert GeoJSON to Shapely geometry
            geom = shape(aoi_geojson)
            
            # Count current vertices (handles Polygon, MultiPolygon, etc.)
            def count_vertices(g):
                """Recursively count vertices in a geometry."""
                if hasattr(g, 'exterior'):
                    # Polygon - count exterior and interior rings
                    count = len(g.exterior.coords)
                    if hasattr(g, 'interiors'):
                        count += sum(len(ring.coords) for ring in g.interiors)
                    return count
                elif hasattr(g, 'geoms'):
                    # MultiPolygon or GeometryCollection - sum all geometries
                    return sum(count_vertices(sub_geom) for sub_geom in g.geoms)
                elif hasattr(g, 'coords'):
                    # LineString, Point, etc.
                    return len(g.coords)
                else:
                    return 0
            
            current_vertices = count_vertices(geom)
            
            print(f"   📐 AOI geometry: {current_vertices} vertices")
            
            # If already simple enough, return as-is
            if current_vertices <= max_vertices:
                print(f"   ✅ Geometry is simple enough ({current_vertices} <= {max_vertices} vertices)")
                return aoi_geojson
            
            # Simplify geometry using Douglas-Peucker algorithm
            # Start with a tolerance and increase until we get below max_vertices
            tolerance = 0.0001  # Start with small tolerance (degrees)
            simplified = geom
            
            while current_vertices > max_vertices and tolerance < 0.01:  # Max tolerance 0.01 degrees (~1km)
                simplified = geom.simplify(tolerance, preserve_topology=True)
                
                # Recalculate vertices using the same function
                current_vertices = count_vertices(simplified)
                
                if current_vertices > max_vertices:
                    tolerance *= 2  # Increase tolerance
                    print(f"   🔄 Simplifying with tolerance {tolerance:.6f}... ({current_vertices} vertices)")
                else:
                    break  # Found good tolerance
            
            # Convert back to GeoJSON
            simplified_geojson = mapping(simplified)
            print(f"   ✅ Simplified to {current_vertices} vertices (tolerance: {tolerance:.6f})")
            
            return simplified_geojson
            
        except ImportError:
            print("⚠️  Shapely not available. Cannot simplify geometry.")
            print("   Consider installing: pip install shapely")
            return aoi_geojson
        except Exception as e:
            print(f"⚠️  Error simplifying geometry: {e}")
            print("   Using original geometry (may fail if too complex)")
            return aoi_geojson
    
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
        
        # Add AOI to map
        # Note: gdf_aoi should already be simplified (from run_complete_workflow)
        # Convert to GeoJSON for Folium
        aoi_geojson = json.loads(gdf_aoi.to_json())
        # Simplify as safety check (will return as-is if already simple)
        simplified_aoi_geojson = self.simplify_aoi_geometry(aoi_geojson, max_vertices=1000)
        folium.GeoJson(
            simplified_aoi_geojson,  # Uses simplified geometry (matches order)
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
    
    def get_items_by_date(self, items: List[Dict], date_str: str) -> List[Dict]:
        """
        Get all items from a specific date.
        Uses the same logic as group_items_by_date() for consistency with map visualization.
        
        Args:
            items: List of item dictionaries
            date_str: Date string in format 'YYYY-MM-DD' (matches map layer names like "Planet Mosaic - 2024-09-03")
            
        Returns:
            List of items from that date
        """
        # Use the same grouping logic as the map visualization
        items_by_date = self.group_items_by_date(items)
        return items_by_date.get(date_str, [])
    
    def get_available_dates(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Get all available dates and their items (same as map visualization).
        Useful for listing available dates to order.
        
        Args:
            items: List of item dictionaries
            
        Returns:
            Dictionary mapping date (YYYY-MM-DD) to list of items
        """
        return self.group_items_by_date(items)
    
    def _normalize_bucket_name(self, bucket_name: str) -> str:
        """
        Normalize GCS bucket name by removing gs:// prefix if present.
        
        Args:
            bucket_name: Bucket name with or without gs:// prefix
            
        Returns:
            Bucket name without prefix
        """
        if bucket_name.startswith('gs://'):
            return bucket_name[5:]  # Remove 'gs://' prefix
        return bucket_name
    
    def _get_gcs_credentials(self, gcs_credentials: Optional[str] = None) -> Optional[Dict]:
        """
        Get GCS credentials from provided parameter or file path.
        
        Args:
            gcs_credentials: JSON string, base64-encoded JSON string, or path to JSON file
            
        Returns:
            Credentials dict or None
        """
        import base64
        import json
        
        if not gcs_credentials:
            return None
        
        # Check if it's a file path
        if os.path.exists(gcs_credentials):
            try:
                with open(gcs_credentials, 'r') as f:
                    return json.load(f)
            except Exception as e:
                raise ValueError(f"Could not load credentials from file '{gcs_credentials}': {e}")
        
        # Try as JSON string first
        try:
            return json.loads(gcs_credentials)
        except json.JSONDecodeError:
            # Try as base64-encoded JSON
            try:
                # Add padding if needed
                padding = len(gcs_credentials) % 4
                if padding:
                    gcs_credentials = gcs_credentials + '=' * (4 - padding)
                creds_json = base64.b64decode(gcs_credentials).decode()
                return json.loads(creds_json)
            except:
                raise ValueError("Invalid credentials format. Expected JSON string, base64-encoded JSON, or path to JSON file")
    
    def verify_gcs_delivery(
        self,
        gcs_bucket: str,
        gcs_credentials: str,
        gcs_path: Optional[str] = None,
        check_existing: bool = True
    ) -> Dict[str, bool]:
        """
        Verify that Planet API can deliver to the specified GCS bucket.
        Tests bucket access and write permissions only (no delete needed).
        Optionally checks if files already exist at the target path.
        
        Args:
            gcs_bucket: GCS bucket name (with or without gs:// prefix, e.g., 'my-bucket' or 'gs://my-bucket')
            gcs_credentials: Path to service account JSON file, JSON string, or base64-encoded JSON string
            gcs_path: Optional GCS path prefix to check for existing files (e.g., 'planet_orders/2024-09-03/')
            check_existing: If True, raise ValueError if files exist at gcs_path
            
        Returns:
            Dictionary with verification results:
            {
                'bucket_exists': bool,
                'can_write': bool,
                'path_exists': bool,  # True if files exist at gcs_path
                'credentials_valid': bool,
                'ready_for_delivery': bool
            }
            
        Raises:
            ValueError: If check_existing=True and files exist at gcs_path
        """
        import base64
        import json
        from google.cloud import storage
        from google.cloud.exceptions import NotFound, Forbidden
        
        results = {
            'bucket_exists': False,
            'can_write': False,
            'path_exists': False,
            'credentials_valid': False,
            'ready_for_delivery': False
        }
        
        # Normalize bucket name (strip gs:// prefix if present)
        gcs_bucket = self._normalize_bucket_name(gcs_bucket)
        
        print(f"\n🔍 Verifying GCS delivery setup for bucket: {gcs_bucket}")
        
        # Step 1: Load and verify credentials
        try:
            creds = self._get_gcs_credentials(gcs_credentials)
            if creds is None:
                print(f"   ❌ No credentials found")
                return results
            
            results['credentials_valid'] = True
            if os.path.exists(gcs_credentials):
                print(f"   ✅ Credentials loaded from file: {gcs_credentials}")
            else:
                print("   ✅ Credentials provided (JSON string or base64)")
            
            # Step 2: Test bucket access
            try:
                # Create storage client with credentials
                from google.oauth2 import service_account
                creds_obj = service_account.Credentials.from_service_account_info(
                    creds,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                project_id = creds.get('project_id') or self.gcs_project
                client = storage.Client(credentials=creds_obj, project=project_id)
                
                # Check if bucket exists
                try:
                    bucket = client.bucket(gcs_bucket)
                    bucket.reload()  # This will raise NotFound if bucket doesn't exist
                    results['bucket_exists'] = True
                    print(f"   ✅ Bucket exists: {gcs_bucket}")
                except NotFound:
                    print(f"   ❌ Bucket does not exist: {gcs_bucket}")
                    return results
                except Forbidden:
                    print(f"   ❌ Access denied to bucket: {gcs_bucket}")
                    return results
                
                # Step 3: Test write permission (Planet needs to write files)
                # Note: We only test write/create permission, NOT delete permission
                # Test in the main directory from gcs_path to respect IAM path restrictions
                try:
                    import time
                    # Use timestamped filename to avoid conflicts
                    test_filename = f"planetverify_{int(time.time())}.txt"
                    
                    # If gcs_path is provided, extract main directory and test write permission there
                    # This ensures the test respects IAM conditions that restrict to specific paths
                    # e.g., 'korindo/2024-09-03/' -> main dir is 'korindo/'
                    if gcs_path:
                        # Extract main directory (first part before second slash)
                        path_parts = gcs_path.strip('/').split('/')
                        if path_parts and path_parts[0]:
                            main_dir = path_parts[0] + '/'
                            test_blob_name = f"{main_dir}{test_filename}"
                            print(f"   🔍 Testing write permission in main directory: {main_dir}")
                        else:
                            test_blob_name = test_filename
                            print(f"   🔍 Testing write permission in bucket root")
                    else:
                        test_blob_name = test_filename
                        print(f"   🔍 Testing write permission in bucket root")
                    
                    test_blob = bucket.blob(test_blob_name)
                    test_blob.upload_from_string("Planet API delivery test - can be safely deleted")
                    results['can_write'] = True
                    print(f"   ✅ Write permission: OK")
                    print(f"   ℹ️  Test file created: {test_blob_name} (can be manually deleted if needed)")
                    
                except Forbidden as e:
                    print(f"   ❌ Write permission: DENIED")
                    print(f"   Error details: {e}")
                    service_account_email = creds.get('client_email', 'Unknown')
                    print(f"\n   💡 To fix this, grant the following IAM role to service account:")
                    print(f"      Service Account: {service_account_email}")
                    print(f"      Role: Storage Object Creator (roles/storage.objectCreator)")
                    print(f"      Or grant these permissions:")
                    print(f"        - storage.objects.create")
                    print(f"        - storage.objects.get")
                    print(f"      On bucket: {gcs_bucket}")
                    print(f"\n   Command to grant access:")
                    print(f"      gsutil iam ch serviceAccount:{service_account_email}:roles/storage.objectCreator gs://{gcs_bucket}")
                    return results
                except Exception as e:
                    print(f"   ⚠️  Error testing write permission: {e}")
                    import traceback
                    traceback.print_exc()
                    return results
                
                # Step 4: Check if files already exist at target path (if path provided)
                if gcs_path:
                    path_prefix = gcs_path.rstrip('/') + '/'
                    print(f"\n   🔍 Checking for existing files at: gs://{gcs_bucket}/{path_prefix}")
                    try:
                        # List blobs with the path prefix
                        blobs = list(bucket.list_blobs(prefix=path_prefix, max_results=1))
                        if blobs:
                            results['path_exists'] = True
                            existing_count = len(list(bucket.list_blobs(prefix=path_prefix)))
                            print(f"   ⚠️  Found {existing_count} existing file(s) at target path")
                            
                            if check_existing:
                                raise ValueError(
                                    f"Files already exist at gs://{gcs_bucket}/{path_prefix}\n"
                                    f"Found {existing_count} file(s). Please:\n"
                                    f"  1. Delete existing files, or\n"
                                    f"  2. Use a different gcs_path, or\n"
                                    f"  3. Set check_existing=False to proceed anyway"
                                )
                            else:
                                print(f"   ⚠️  Proceeding despite existing files (check_existing=False)")
                        else:
                            print(f"   ✅ No existing files at target path")
                    except ValueError:
                        raise  # Re-raise ValueError from check_existing
                    except Exception as e:
                        print(f"   ⚠️  Could not check for existing files: {e}")
                        # Don't fail verification if we can't check - just warn
                
            except Exception as e:
                print(f"   ❌ Error accessing GCS: {e}")
                return results
            
            # Final check (only need write permission, not delete)
            results['ready_for_delivery'] = (
                results['bucket_exists'] and 
                results['can_write'] and 
                results['credentials_valid'] and
                not (check_existing and results['path_exists'])  # Not ready if files exist and check_existing=True
            )
            
            if results['ready_for_delivery']:
                path_info = f"gs://{gcs_bucket}/{gcs_path}" if gcs_path else f"gs://{gcs_bucket}/"
                print(f"\n✅ GCS delivery is ready!")
                print(f"   Planet API can deliver files to: {path_info}")
            else:
                print(f"\n⚠️  GCS delivery may not work:")
                if not results['bucket_exists']:
                    print(f"   - Bucket does not exist")
                if not results['can_write']:
                    print(f"   - Cannot write to bucket")
                if check_existing and results['path_exists']:
                    print(f"   - Files already exist at target path (use different path or delete existing files)")
                if not results['credentials_valid']:
                    print(f"   - Credentials invalid")
            
            return results
            
        except Exception as e:
            print(f"   ❌ Error during verification: {e}")
            import traceback
            traceback.print_exc()
            return results
    
    def create_clipped_harmonized_order(
        self,
        item_ids: List[str],
        aoi_geojson: Dict,
        bundle_type: str = 'analytic_8b_sr_udm2',
        order_name: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        gcs_path: Optional[str] = None,
        gcs_credentials: Optional[str] = None,
        verify_gcs: bool = True
    ) -> Dict:
        """
        Create an order for harmonized 8-band PlanetScope mosaic clipped to AOI.
        Includes: Clip, Composite (mosaic), Harmonize (Sentinel-2), COG format, and UDM.
        Matches successful order structure from Planet API.
        
        Args:
            item_ids: List of item IDs to include in mosaic
            aoi_geojson: GeoJSON geometry for clipping
            bundle_type: Bundle type (default: 'analytic_8b_sr_udm2' for 8-band harmonized + UDM)
            order_name: Optional order name
            gcs_bucket: GCS bucket name for delivery (optional, with or without gs:// prefix)
            gcs_path: GCS path prefix/folder (optional, e.g., 'planet_orders/2024-09-03/')
            gcs_credentials: Path to service account JSON file, JSON string, or base64-encoded JSON string
                           (required if gcs_bucket is provided)
            
        Returns:
            Order details dictionary
        """
        # Prepare GCS delivery if bucket is provided
        gcs_delivery = None
        if gcs_bucket:
            # Normalize bucket name (strip gs:// prefix if present)
            gcs_bucket = self._normalize_bucket_name(gcs_bucket)
            
            if not gcs_credentials:
                raise ValueError(
                    "gcs_credentials is required when gcs_bucket is provided. "
                    "Provide path to service account JSON file, JSON string, or base64-encoded JSON."
                )
            
            # Verify GCS access before creating order (if requested)
            if verify_gcs:
                try:
                    verification = self.verify_gcs_delivery(
                        gcs_bucket, 
                        gcs_credentials,
                        gcs_path=gcs_path,
                        check_existing=True  # Raise error if files exist
                    )
                    if not verification['ready_for_delivery']:
                        print(f"\n⚠️  WARNING: GCS delivery verification failed!")
                        print(f"   Order will still be created, but delivery may fail.")
                        print(f"   Please fix GCS permissions before creating orders with delivery.")
                        # Don't block order creation, just warn
                except ValueError as e:
                    # Re-raise ValueError (e.g., files exist) - this should block order creation
                    raise
            
            # Prepare delivery configuration
            gcs_delivery = {"bucket": gcs_bucket}
            if gcs_path:
                gcs_delivery["path_prefix"] = gcs_path.rstrip('/') + '/'
            
            # Get credentials and encode to base64 for Planet API
            import base64
            import json
            
            try:
                # Get credentials using the helper method (handles JSON string, base64, or file path)
                creds = self._get_gcs_credentials(gcs_credentials)
                if creds is None:
                    raise ValueError("Could not load GCS credentials")
                
                # Verify credentials have required fields
                if not isinstance(creds, dict):
                    raise ValueError("Credentials must be a dictionary")
                if 'type' not in creds or creds.get('type') != 'service_account':
                    raise ValueError("Credentials must be a service account JSON")
                if 'private_key' not in creds:
                    raise ValueError("Credentials missing 'private_key' field")
                
                # SECURITY NOTE: Planet API requires the full service account JSON
                # including the private_key to authenticate and write to GCS.
                # This is standard for GCS direct delivery - Planet needs to impersonate
                # your service account to write files. The credentials are base64-encoded
                # and sent over HTTPS to Planet's secure API.
                # 
                # Best practices:
                # 1. Use a dedicated service account with minimal permissions
                #    (only Storage Object Admin on the specific bucket)
                # 2. Rotate credentials periodically
                # 3. Monitor service account usage in GCP audit logs
                
                # Encode to base64 for Planet API
                creds_json = json.dumps(creds)
                creds_base64 = base64.b64encode(creds_json.encode()).decode()
                gcs_delivery["credentials"] = creds_base64
                
                if os.path.exists(gcs_credentials):
                    print(f"   ✅ Credentials loaded from file and encoded to base64")
                else:
                    print(f"   ✅ Credentials encoded to base64 for Planet API")
                print(f"   🔒 Note: Full service account JSON (including private_key) will be sent to Planet API")
                print(f"      This is required for GCS direct delivery. Use a dedicated service account with minimal permissions.")
                print(f"   📋 GCS delivery config: bucket={gcs_bucket}, path_prefix={gcs_delivery.get('path_prefix', 'N/A')}, credentials={'***' if creds_base64 else 'MISSING'}")
            except Exception as e:
                raise ValueError(f"Could not process GCS credentials: {e}")
        
        # Simplify AOI geometry if needed (Planet API limit: 1500 vertices)
        simplified_aoi = self.simplify_aoi_geometry(aoi_geojson, max_vertices=1000)
        
        return self.downloader.create_order_with_clip(
            item_ids=item_ids,
            aoi_geometry=simplified_aoi,  # Use simplified geometry
            bundle_type=bundle_type,
            order_name=order_name,
            harmonized=True,
            gcs_delivery=gcs_delivery
        )
    
    def check_order_status(self, order_id: str) -> Dict:
        """
        Check order status without waiting.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order details dictionary with current state
        """
        return self.downloader.get_order_status(order_id)
    
    def download_order_to_gcs(
        self,
        order_id: str,
        gcs_path: str,
        extract_zip: bool = True
    ) -> List[str]:
        """
        Download order files directly to Google Cloud Storage.
        Extracts zip files and uploads individual files.
        
        Args:
            order_id: Order ID
            gcs_path: GCS path (e.g., 'gs://bucket/path/')
            extract_zip: If True, extract zip files and upload individual files
            
        Returns:
            List of GCS paths to uploaded files
        """
        return self.downloader.download_order_to_gcs(order_id, gcs_path, extract_zip)
    
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
        aoi_gdf: Optional['gpd.GeoDataFrame'] = None,
        gcs_shp_path: Optional[str] = None,
        gcs_credentials: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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
            aoi_gdf: GeoDataFrame with AOI geometry (optional, if provided, gcs_shp_path is ignored)
            gcs_shp_path: GCS path to shapefile (optional, required if aoi_gdf not provided)
            gcs_credentials: Path to service account JSON file for loading from GCS (optional)
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
        
        # Step 1: Load or use AOI
        print("=" * 60)
        if aoi_gdf is not None:
            print("STEP 1: Using provided AOI GeoDataFrame")
            print("=" * 60)
            gdf_aoi = aoi_gdf.copy()
            # Ensure it's in WGS84 for Planet API
            if gdf_aoi.crs != 'EPSG:4326':
                print(f"🔄 Converting CRS from {gdf_aoi.crs} to EPSG:4326...")
                gdf_aoi = gdf_aoi.to_crs('EPSG:4326')
        elif gcs_shp_path:
            print("STEP 1: Loading AOI from GCS")
            print("=" * 60)
            gdf_aoi = self.load_aoi_from_gcs(gcs_shp_path, gcs_credentials=gcs_credentials)
        else:
            raise ValueError("Either 'aoi_gdf' or 'gcs_shp_path' must be provided")
        
        results['aoi'] = gdf_aoi
        
        # Step 2: Convert to GeoJSON and simplify
        print("\n" + "=" * 60)
        print("STEP 2: Converting AOI to GeoJSON and Simplifying")
        print("=" * 60)
        aoi_geojson = self.aoi_to_geojson(gdf_aoi)
        # Simplify AOI geometry (Planet API limit: 1500 vertices)
        # This ensures visualization matches what will be used in orders
        simplified_aoi_geojson = self.simplify_aoi_geometry(aoi_geojson, max_vertices=1000)
        results['aoi_geojson'] = simplified_aoi_geojson  # Store simplified version
        results['aoi_geojson_original'] = aoi_geojson  # Keep original for reference
        
        # Step 3: Search images
        if not start_date or not end_date:
            raise ValueError("start_date and end_date are required")
        
        print("\n" + "=" * 60)
        print("STEP 3: Searching PlanetScope Images")
        print("=" * 60)
        items = self.search_and_filter(
            aoi_geojson=simplified_aoi_geojson,  # Use simplified for search too
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
            # Create simplified GeoDataFrame for visualization (matches order geometry)
            from shapely.geometry import shape
            simplified_gdf = gpd.GeoDataFrame(
                [1], 
                geometry=[shape(simplified_aoi_geojson)], 
                crs=gdf_aoi.crs
            )
            map_obj = self.visualize_results(simplified_gdf, items, add_tile_layers=True)
            results['map'] = map_obj
            print("\n💡 Use the layer control (top-right) to toggle date-based mosaic previews!")
            print("   📐 Note: AOI shown in map is simplified (matches order geometry)")
        
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


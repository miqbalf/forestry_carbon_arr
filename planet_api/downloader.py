"""
PlanetScope Image Downloader using Planet Labs API v3.4.0
Replicates web UI functionality for searching, activating, and downloading images
"""

import os
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import requests
from planet import Session
from planet import data_filter

from .utils import parse_date, run_async


class PlanetScopeDownloader:
    """
    Download PlanetScope images using Planet Labs API v3.4.0.
    Provides same functionality as Planet web UI.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Planet API client.
        
        Args:
            api_key: Planet API key. If None, reads from PLANET_API_KEY env var.
        """
        # Use PLANET_API_KEY as primary reference
        self.api_key = api_key or os.getenv('PLANET_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Set PLANET_API_KEY env var "
                "or pass api_key parameter."
            )
        
        # Set PLANET_API_KEY as primary
        os.environ['PLANET_API_KEY'] = self.api_key
        
        # If Planet SDK expects PL_API_KEY, set it from PLANET_API_KEY
        if not os.getenv('PL_API_KEY'):
            os.environ['PL_API_KEY'] = self.api_key
        
    def search_images(
        self,
        geometry: Dict,  # GeoJSON geometry
        start_date: str,  # Format: "2024-01-01T00:00:00Z"
        end_date: str,    # Format: "2024-12-31T23:59:59Z"
        item_type: str = 'PSScene',
        cloud_cover_max: float = 0.1,  # 10% max cloud cover
        limit: int = 100
    ) -> List[Dict]:
        """
        Search for PlanetScope images matching criteria.
        Equivalent to web UI search functionality.
        
        Args:
            geometry: GeoJSON geometry dict (Polygon/MultiPolygon)
            start_date: Start date in ISO format
            end_date: End date in ISO format
            item_type: Item type (PSScene, PSScene4Band)
            cloud_cover_max: Maximum cloud cover (0.0-1.0)
            limit: Maximum number of results
            
        Returns:
            List of item dictionaries
        """
        print(f"🔍 Searching for PlanetScope images...")
        print(f"   Date range: {start_date} to {end_date}")
        print(f"   Max cloud cover: {cloud_cover_max*100}%")
        
        # Run async search (handles Jupyter event loop)
        return run_async(self._async_search_images(
            geometry, start_date, end_date, item_type, cloud_cover_max, limit
        ))
    
    async def _async_search_images(
        self,
        geometry: Dict,
        start_date: str,
        end_date: str,
        item_type: str,
        cloud_cover_max: float,
        limit: int
    ) -> List[Dict]:
        """Async implementation of search."""
        # Build filters using data_filter module
        geometry_filter = data_filter.geometry_filter(geometry)
        
        # Convert string dates to datetime objects (required by date_range_filter)
        start_dt = parse_date(start_date)
        end_dt = parse_date(end_date)
        
        # Use date_range_filter with gte/lte parameters (expects datetime objects)
        date_range_filter = data_filter.date_range_filter(
            'acquired', 
            gte=start_dt, 
            lte=end_dt
        )
        
        # Cloud cover filter
        cloud_cover_filter = data_filter.range_filter(
            'cloud_cover', 
            lte=cloud_cover_max
        )
        
        # Combine filters
        combined_filter = data_filter.and_filter([
            geometry_filter,
            date_range_filter,
            cloud_cover_filter
        ])
        
        # Execute search using async API
        # Note: client.search() takes item_types and search_filter as parameters
        # It returns an async generator - iterate directly, don't await
        async with Session() as session:
            client = session.client('data')
            # search() signature: search(item_types, search_filter=..., ...)
            items = []
            count = 0
            async for item in client.search(
                item_types=[item_type],
                search_filter=combined_filter
            ):
                items.append(item)
                count += 1
                if count >= limit:
                    break
            
        print(f"✅ Found {len(items)} images")
        return items
    
    def get_item_assets(self, item_id: str) -> Dict:
        """
        Get available assets for an item.
        Equivalent to viewing asset details in web UI.
        
        Args:
            item_id: Planet item ID
            
        Returns:
            Dictionary of available assets
        """
        return run_async(self._async_get_item_assets(item_id))
    
    async def _async_get_item_assets(self, item_id: str) -> Dict:
        """Async implementation of get_item_assets."""
        async with Session() as session:
            client = session.client('data')
            # Use list_item_assets() which returns an async iterator
            assets = {}
            async for asset in client.list_item_assets(item_id):
                asset_type = asset.get('type') or asset.get('name', 'unknown')
                assets[asset_type] = asset
            return assets
    
    def activate_asset(self, item_id: str, asset_type: str) -> bool:
        """
        Activate an asset for download.
        Equivalent to clicking "Activate" in web UI.
        
        Args:
            item_id: Planet item ID
            asset_type: Asset type (e.g., 'analytic_sr_udm2')
            
        Returns:
            True if activation successful
        """
        return run_async(self._async_activate_asset(item_id, asset_type))
    
    async def _async_activate_asset(self, item_id: str, asset_type: str) -> bool:
        """Async implementation of activate_asset."""
        async with Session() as session:
            client = session.client('data')
            
            # Get assets using list_item_assets (returns async iterator)
            assets = {}
            async for asset in client.list_item_assets(item_id):
                asset_type_key = asset.get('type') or asset.get('name', 'unknown')
                assets[asset_type_key] = asset
            
            if asset_type not in assets:
                print(f"❌ Asset type '{asset_type}' not available for {item_id}")
                return False
            
            asset = assets[asset_type]
            
            # Check if already active
            if asset.get('status') == 'active':
                print(f"✅ Asset already active: {item_id}/{asset_type}")
                return True
            
            # Activate asset using get_asset and activate_asset
            print(f"🔄 Activating asset: {item_id}/{asset_type}")
            asset_obj = await client.get_asset(item_id, asset_type)
            await client.activate_asset(item_id, asset_type)
            
            # Wait for activation
            max_wait = 300  # 5 minutes max
            wait_time = 0
            while wait_time < max_wait:
                await asyncio.sleep(2)
                wait_time += 2
                
                # Get assets using list_item_assets
                assets = {}
                async for asset in client.list_item_assets(item_id):
                    asset_type_key = asset.get('type') or asset.get('name', 'unknown')
                    assets[asset_type_key] = asset
                
                asset = assets.get(asset_type)
                if not asset:
                    print(f"❌ Asset not found: {item_id}/{asset_type}")
                    return False
                
                status = asset.get('status')
                if status == 'active':
                    print(f"✅ Asset activated: {item_id}/{asset_type}")
                    return True
                elif status == 'failed':
                    print(f"❌ Activation failed: {item_id}/{asset_type}")
                    return False
                
                if wait_time % 10 == 0:
                    print(f"   Still waiting... ({wait_time}s)")
            
            print(f"⏰ Timeout waiting for activation")
            return False
    
    def download_asset(
        self, 
        item_id: str, 
        asset_type: str, 
        output_dir: str = './downloads',
        chunk_size: int = 8192
    ) -> Optional[str]:
        """
        Download an asset.
        Equivalent to clicking "Download" in web UI.
        
        Args:
            item_id: Planet item ID
            asset_type: Asset type to download
            output_dir: Output directory for downloads
            chunk_size: Download chunk size
            
        Returns:
            Path to downloaded file, or None if failed
        """
        # Ensure asset is activated
        if not self.activate_asset(item_id, asset_type):
            return None
        
        # Download using async method
        return run_async(self._async_download_asset(item_id, asset_type, output_dir, chunk_size))
    
    async def _async_download_asset(
        self,
        item_id: str,
        asset_type: str,
        output_dir: str,
        chunk_size: int
    ) -> Optional[str]:
        """Async implementation of download_asset."""
        # Get asset download URL using get_asset method
        async with Session() as session:
            client = session.client('data')
            asset = await client.get_asset(item_id, asset_type)
        
        if not asset or asset.get('status') != 'active':
            print(f"❌ Asset not active: {item_id}/{asset_type}")
            return None
        
        # Get download URL
        download_url = asset.get('location')
        if not download_url:
            print(f"❌ No download URL available: {item_id}/{asset_type}")
            return None
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{item_id}_{asset_type}.tif"
        filepath = os.path.join(output_dir, filename)
        
        # Download file
        print(f"⬇️  Downloading: {filename}")
        response = requests.get(download_url, stream=True, headers={
            'Authorization': f'api-key {self.api_key}'
        })
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded: {filepath}")
        return filepath
    
    def create_order(
        self,
        item_ids: List[str],
        bundle_type: str = 'analytic_sr_udm2',
        order_name: Optional[str] = None
    ) -> Dict:
        """
        Create an order for bulk download.
        Equivalent to creating an order in web UI.
        
        Args:
            item_ids: List of item IDs
            bundle_type: Bundle type ('visual', 'analytic', 'analytic_sr_udm2')
            order_name: Optional order name
            
        Returns:
            Order details dictionary
        """
        return run_async(self._async_create_order(item_ids, bundle_type, order_name))
    
    async def _async_create_order(
        self,
        item_ids: List[str],
        bundle_type: str,
        order_name: Optional[str]
    ) -> Dict:
        """Async implementation of create_order."""
        if order_name is None:
            order_name = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📦 Creating order: {order_name}")
        print(f"   Items: {len(item_ids)}")
        print(f"   Bundle type: {bundle_type}")
        
        # Build order request
        order_request = {
            "name": order_name,
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": "PSScene",
                    "product_bundle": bundle_type
                }
            ]
        }
        
        # Create order using async API
        async with Session() as session:
            client = session.client('orders')
            order = await client.create_order(order_request)
            order_id = order['id']
            print(f"✅ Order created: {order_id}")
            return order
    
    def wait_for_order(self, order_id: str, timeout: int = 3600) -> Dict:
        """
        Wait for order to complete.
        
        Args:
            order_id: Order ID
            timeout: Maximum wait time in seconds
            
        Returns:
            Order details when complete
        """
        return run_async(self._async_wait_for_order(order_id, timeout))
    
    async def _async_wait_for_order(self, order_id: str, timeout: int) -> Dict:
        """Async implementation of wait_for_order."""
        print(f"⏳ Waiting for order to complete: {order_id}")
        start_time = time.time()
        
        async with Session() as session:
            client = session.client('orders')
            
            while time.time() - start_time < timeout:
                order = await client.get_order(order_id)
                state = order['state']
                
                print(f"   Order state: {state}")
                
                if state == 'success':
                    print(f"✅ Order completed successfully!")
                    return order
                elif state == 'failed':
                    print(f"❌ Order failed!")
                    return order
                
                await asyncio.sleep(10)
            
            print(f"⏰ Timeout waiting for order")
            return await client.get_order(order_id)
    
    def download_order(self, order_id: str, output_dir: str = './downloads') -> List[str]:
        """
        Download all files from a completed order.
        
        Args:
            order_id: Order ID
            output_dir: Output directory
            
        Returns:
            List of downloaded file paths
        """
        return run_async(self._async_download_order(order_id, output_dir))
    
    async def _async_download_order(self, order_id: str, output_dir: str) -> List[str]:
        """Async implementation of download_order."""
        async with Session() as session:
            client = session.client('orders')
            order = await client.get_order(order_id)
        
        if order['state'] != 'success':
            print(f"❌ Order not ready for download. State: {order['state']}")
            return []
        
        print(f"⬇️  Downloading order files...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        # Get download links from order
        results = order.get('_links', {}).get('results', [])
        
        for delivery in results:
            url = delivery.get('location')
            filename = delivery.get('name', f"file_{len(downloaded_files)}.zip")
            filepath = os.path.join(output_dir, filename)
            
            print(f"   Downloading: {filename}")
            response = requests.get(
                url, 
                stream=True,
                headers={'Authorization': f'api-key {self.api_key}'}
            )
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            downloaded_files.append(filepath)
            print(f"   ✅ Downloaded: {filename}")
        
        print(f"✅ Downloaded {len(downloaded_files)} files")
        return downloaded_files
    
    def create_order_with_clip(
        self,
        item_ids: List[str],
        aoi_geometry: Dict,
        bundle_type: str = 'analytic_sr_udm2',
        order_name: Optional[str] = None,
        harmonized: bool = True
    ) -> Dict:
        """
        Create an order with clip tool to AOI.
        Supports harmonized 8-band PlanetScope data.
        
        Args:
            item_ids: List of item IDs
            aoi_geometry: GeoJSON geometry for clipping
            bundle_type: Bundle type ('visual', 'analytic', 'analytic_sr_udm2')
            order_name: Optional order name
            harmonized: If True, use harmonized 8-band data
            
        Returns:
            Order details dictionary
        """
        return run_async(self._async_create_order_with_clip(
            item_ids, aoi_geometry, bundle_type, order_name, harmonized
        ))
    
    async def _async_create_order_with_clip(
        self,
        item_ids: List[str],
        aoi_geometry: Dict,
        bundle_type: str,
        order_name: Optional[str],
        harmonized: bool
    ) -> Dict:
        """Async implementation of create_order_with_clip."""
        if order_name is None:
            order_name = f"order_clipped_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📦 Creating clipped order: {order_name}")
        print(f"   Items: {len(item_ids)}")
        print(f"   Bundle type: {bundle_type}")
        print(f"   Harmonized: {harmonized}")
        
        # Build order request with clip, harmonize, and COG format tools
        tools = [
            {
                "clip": {
                    "aoi": aoi_geometry
                }
            }
        ]
        
        # Add harmonization and COG format tools
        if harmonized:
            # Add harmonize tool (target sensor: Sentinel-2)
            tools.append({
                "harmonize": {
                    "target_sensor": "Sentinel-2"
                }
            })
            
            # Add file format tool for COG
            tools.append({
                "file_format": {
                    "format": "COG"
                }
            })
            
            print("   ✅ Added harmonization (Sentinel-2) and COG format tools")
        
        order_request = {
            "name": order_name,
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": "PSScene",
                    "product_bundle": bundle_type
                }
            ],
            "tools": tools
        }
        
        # Create order using async API
        async with Session() as session:
            client = session.client('orders')
            order = await client.create_order(order_request)
            order_id = order['id']
            print(f"✅ Order created: {order_id}")
            return order
    
    def get_item_thumbnail_url(self, item_id: str) -> Optional[str]:
        """
        Get thumbnail/preview URL for an item.
        Useful for visualization before ordering.
        
        Args:
            item_id: Planet item ID
            
        Returns:
            Thumbnail URL or None
        """
        return run_async(self._async_get_item_thumbnail_url(item_id))
    
    async def _async_get_item_thumbnail_url(self, item_id: str) -> Optional[str]:
        """Async implementation of get_item_thumbnail_url."""
        async with Session() as session:
            client = session.client('data')
            item = await client.get_item('PSScene', item_id)
            
            # Check for thumbnail in _links
            links = item.get('_links', {})
            thumbnail_url = links.get('thumbnail')
            
            if thumbnail_url:
                return thumbnail_url
            else:
                # Try alternative locations
                properties = item.get('properties', {})
                thumbnail = properties.get('thumbnail')
                return thumbnail
    
    def convert_to_cog(
        self,
        input_path: str,
        output_path: str,
        temp_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Convert GeoTIFF to Cloud Optimized GeoTIFF (COG) using GDAL.
        
        Args:
            input_path: Path to input GeoTIFF
            output_path: Path to output COG
            temp_dir: Temporary directory for processing (default: /mnt/data)
            
        Returns:
            Path to created COG file, or None if failed
        """
        try:
            from osgeo import gdal
            import os
            
            # Set temp directory
            if temp_dir is None:
                temp_dir = '/mnt/data'
            
            os.makedirs(temp_dir, exist_ok=True)
            os.environ['CPL_TMPDIR'] = temp_dir
            os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'
            
            # Optimize GDAL settings
            gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
            
            print(f"🔄 Converting to COG...")
            print(f"   Input: {input_path}")
            print(f"   Output: {output_path}")
            
            # COG creation options
            translate_options = [
                "COMPRESS=ZSTD",
                "BIGTIFF=YES",
                "PREDICTOR=2",
                "BLOCKSIZE=512",
                "OVERVIEWS=AUTO",
                "NUM_THREADS=ALL_CPUS"
            ]
            
            # Convert to COG
            result = gdal.Translate(
                output_path,
                input_path,
                format="COG",
                creationOptions=translate_options
            )
            
            if result is not None:
                print(f"✅ COG created: {output_path}")
                return output_path
            else:
                print(f"❌ COG conversion failed")
                return None
                
        except ImportError:
            print("❌ GDAL not available. Install with: pip install gdal")
            return None
        except Exception as e:
            print(f"❌ Error converting to COG: {e}")
            return None


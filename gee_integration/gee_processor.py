"""
Google Earth Engine processor for Forestry Carbon ARR analysis.
"""

import logging
from typing import Dict, Any, Optional, Union, List
import ee
import geemap

from ..exceptions import GEEError, DependencyError
from ..utils.dependency_manager import DependencyManager

logger = logging.getLogger(__name__)


class GEEProcessor:
    """
    Google Earth Engine processor for forestry carbon analysis.
    
    This class provides a high-level interface to GEE operations,
    integrating with the GEE_notebook_Forestry modules when available.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GEE processor.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.config = config
        self.dependency_manager = DependencyManager()
        
        # Check if GEE dependencies are available
        if not self.dependency_manager.is_gee_available():
            raise DependencyError("GEE dependencies not available. Please install earthengine-api and geemap.")
        
        # Initialize GEE
        self._initialize_gee()
        
        # Try to import GEE Forestry modules
        self._gee_forestry_modules = self._import_gee_forestry_modules()
    
    def _initialize_gee(self) -> None:
        """Initialize Google Earth Engine."""
        try:
            gee_config = self.config.get('gee', {})
            project_id = gee_config.get('project_id')
            
            if project_id:
                ee.Initialize(project=project_id)
                self.logger.info(f"GEE initialized with project: {project_id}")
            else:
                ee.Initialize()
                self.logger.info("GEE initialized with default project")
                
        except Exception as e:
            raise GEEError(f"Failed to initialize Google Earth Engine: {e}")
    
    def _import_gee_forestry_modules(self) -> Dict[str, Any]:
        """
        Import GEE_notebook_Forestry modules if available.
        
        Returns:
            Dictionary of imported modules
        """
        modules = {}
        
        if not self.dependency_manager.is_gee_forestry_available():
            self.logger.warning("GEE_notebook_Forestry modules not available")
            return modules
        
        try:
            # Import key modules from GEE_notebook_Forestry
            import osi.image_collection.main as image_collection
            import osi.spectral_indices.spectral_analysis as spectral_analysis
            import osi.fcd.main_fcd as fcd_main
            import osi.ml.main as ml_main
            import osi.utils.main as utils_main
            
            modules.update({
                'image_collection': image_collection,
                'spectral_analysis': spectral_analysis,
                'fcd_main': fcd_main,
                'ml_main': ml_main,
                'utils_main': utils_main
            })
            
            self.logger.info("GEE_notebook_Forestry modules imported successfully")
            
        except ImportError as e:
            self.logger.warning(f"Some GEE_notebook_Forestry modules could not be imported: {e}")
        
        return modules
    
    def create_image_collection(self, 
                              satellite_provider: str,
                              date_range: List[str],
                              aoi: ee.Geometry,
                              cloud_cover_threshold: int = 40) -> Any:
        """
        Create image collection using GEE_notebook_Forestry if available.
        
        Args:
            satellite_provider: Satellite provider (Sentinel, Planet, Landsat)
            date_range: Date range [start_date, end_date]
            aoi: Area of interest geometry
            cloud_cover_threshold: Cloud cover threshold percentage
            
        Returns:
            Image collection object
        """
        if 'image_collection' in self._gee_forestry_modules:
            try:
                # Use GEE_notebook_Forestry ImageCollection
                ImageCollection = self._gee_forestry_modules['image_collection'].ImageCollection
                
                collection = ImageCollection(
                    I_satellite=satellite_provider,
                    AOI=aoi,
                    date_start_end=date_range,
                    cloud_cover_threshold=cloud_cover_threshold,
                    region=self.config.get('project', {}).get('region', 'global')
                )
                
                self.logger.info(f"Image collection created using GEE_notebook_Forestry: {satellite_provider}")
                return collection
                
            except Exception as e:
                self.logger.error(f"Failed to create image collection with GEE_notebook_Forestry: {e}")
                raise GEEError(f"Image collection creation failed: {e}")
        else:
            # Fallback to basic GEE implementation
            return self._create_basic_image_collection(satellite_provider, date_range, aoi, cloud_cover_threshold)
    
    def _create_basic_image_collection(self, 
                                     satellite_provider: str,
                                     date_range: List[str],
                                     aoi: ee.Geometry,
                                     cloud_cover_threshold: int) -> ee.ImageCollection:
        """
        Create basic image collection using standard GEE API.
        
        Args:
            satellite_provider: Satellite provider
            date_range: Date range
            aoi: Area of interest
            cloud_cover_threshold: Cloud cover threshold
            
        Returns:
            Basic image collection
        """
        start_date, end_date = date_range
        
        if satellite_provider.lower() == 'sentinel':
            collection = ee.ImageCollection('COPERNICUS/S2_SR') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_threshold))
        
        elif satellite_provider.lower() == 'landsat':
            collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
                .filterDate(start_date, end_date) \
                .filterBounds(aoi) \
                .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover_threshold))
        
        else:
            raise GEEError(f"Unsupported satellite provider: {satellite_provider}")
        
        self.logger.info(f"Basic image collection created: {satellite_provider}")
        return collection
    
    def calculate_forest_canopy_density(self, image: ee.Image) -> Dict[str, ee.Image]:
        """
        Calculate Forest Canopy Density using GEE_notebook_Forestry if available.
        
        Args:
            image: Input satellite image
            
        Returns:
            Dictionary with FCD results
        """
        if 'fcd_main' in self._gee_forestry_modules:
            try:
                # Use GEE_notebook_Forestry FCD calculation
                FCDCalc = self._gee_forestry_modules['fcd_main'].FCDCalc
                fcd_calc = FCDCalc(self.config)
                results = fcd_calc.fcd_calc()
                
                self.logger.info("FCD calculated using GEE_notebook_Forestry")
                return results
                
            except Exception as e:
                self.logger.error(f"Failed to calculate FCD with GEE_notebook_Forestry: {e}")
                raise GEEError(f"FCD calculation failed: {e}")
        else:
            # Fallback to basic NDVI-based approach
            return self._calculate_basic_fcd(image)
    
    def _calculate_basic_fcd(self, image: ee.Image) -> Dict[str, ee.Image]:
        """
        Calculate basic FCD using NDVI.
        
        Args:
            image: Input satellite image
            
        Returns:
            Dictionary with basic FCD results
        """
        # Simple NDVI-based forest density
        ndvi = image.normalizedDifference(['B8', 'B4'])  # Assuming Sentinel-2 bands
        
        # Basic thresholds
        fcd = ndvi.multiply(100).clamp(0, 100)
        
        return {
            'FCD1_1': fcd,
            'FCD2_1': fcd
        }
    
    def perform_landcover_classification(self, 
                                       image: ee.Image,
                                       training_data: ee.FeatureCollection,
                                       algorithm: str = 'gbm') -> ee.Image:
        """
        Perform land cover classification using GEE_notebook_Forestry if available.
        
        Args:
            image: Input satellite image
            training_data: Training data feature collection
            algorithm: ML algorithm to use
            
        Returns:
            Classified image
        """
        if 'ml_main' in self._gee_forestry_modules:
            try:
                # Use GEE_notebook_Forestry ML classification
                LandcoverML = self._gee_forestry_modules['ml_main'].LandcoverML
                
                ml_class = LandcoverML(
                    image=image,
                    training_data=training_data,
                    algorithm=algorithm
                )
                
                classified = ml_class.train_and_classify()
                
                self.logger.info(f"Land cover classification completed using GEE_notebook_Forestry: {algorithm}")
                return classified
                
            except Exception as e:
                self.logger.error(f"Failed to perform classification with GEE_notebook_Forestry: {e}")
                raise GEEError(f"Classification failed: {e}")
        else:
            # Fallback to basic GEE classification
            return self._perform_basic_classification(image, training_data)
    
    def _perform_basic_classification(self, 
                                    image: ee.Image,
                                    training_data: ee.FeatureCollection) -> ee.Image:
        """
        Perform basic land cover classification using standard GEE.
        
        Args:
            image: Input satellite image
            training_data: Training data
            
        Returns:
            Classified image
        """
        # Basic supervised classification
        bands = image.bandNames()
        
        # Train classifier
        classifier = ee.Classifier.smileRandomForest(10)
        trained = classifier.train(training_data, 'class', bands)
        
        # Classify
        classified = image.classify(trained)
        
        self.logger.info("Basic land cover classification completed")
        return classified
    
    def export_image(self, 
                    image: ee.Image,
                    description: str,
                    folder: str = 'forestry_carbon_analysis',
                    region: Optional[ee.Geometry] = None,
                    scale: int = 30) -> ee.batch.Task:
        """
        Export image to Google Drive or Cloud Storage.
        
        Args:
            image: Image to export
            description: Export description
            folder: Export folder name
            region: Export region
            scale: Export scale
            
        Returns:
            Export task
        """
        try:
            if region is None:
                region = image.geometry()
            
            task = ee.batch.Export.image.toDrive(
                image=image,
                description=description,
                folder=folder,
                region=region,
                scale=scale,
                crs=self.config.get('gee', {}).get('crs', 'EPSG:4326')
            )
            
            task.start()
            self.logger.info(f"Export task started: {description}")
            return task
            
        except Exception as e:
            raise GEEError(f"Export failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get GEE processor system information.
        
        Returns:
            Dictionary with system information
        """
        return {
            'gee_initialized': True,
            'gee_forestry_available': len(self._gee_forestry_modules) > 0,
            'available_modules': list(self._gee_forestry_modules.keys()),
            'config': self.config.get('gee', {})
        }

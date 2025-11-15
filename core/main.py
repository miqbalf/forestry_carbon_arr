"""
Main Forestry Carbon ARR class for GEE_notebook_Forestry integration management.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    import xarray as xr

try:
    from ..config import ConfigManager
    from ..exceptions import ForestryCarbonError, DependencyError
    from ..utils.dependency_manager import DependencyManager
    from ..utils.path_resolver import PathResolver
    from ..utils.import_manager import ImportManager
    from .. import __version__
except ImportError:
    # For direct execution
    from config import ConfigManager
    from exceptions import ForestryCarbonError, DependencyError
    from utils.dependency_manager import DependencyManager
    from utils.path_resolver import PathResolver
    from utils.import_manager import ImportManager
    try:
        from forestry_carbon_arr import __version__
    except ImportError:
        __version__ = "0.1.0"

logger = logging.getLogger(__name__)


class ForestryCarbonARR:
    """
    Main class for GEE_notebook_Forestry integration management.
    
    This class provides import and path management for integrating with GEE_notebook_Forestry,
    supporting both local development (side-by-side) and container environments.
    
    Features:
    - Automatic GEE_notebook_Forestry detection and path resolution
    - Support for multiple deployment scenarios (container, standalone, development)
    - Import strategy management (local vs container)
    - Flexible dependency management
    """
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path, Dict]] = None,
                 gee_forestry_path: Optional[Union[str, Path]] = None,
                 auto_setup: bool = True):
        """
        Initialize Forestry Carbon ARR integration system.
        
        Args:
            config_path: Path to configuration file, config dict, or None for defaults
            gee_forestry_path: Path to GEE_notebook_Forestry directory (optional)
            auto_setup: Whether to automatically setup dependencies and paths
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Initialize managers
        self.dependency_manager = DependencyManager()
        self.path_resolver = PathResolver()
        self.import_manager = ImportManager()
        
        # Set GEE Forestry path if provided
        if gee_forestry_path:
            self.set_gee_forestry_path(gee_forestry_path)
        
        # Integration status
        self._gee_forestry_available = False
        self._gee_forestry_path = None
        self._import_strategy = None
        
        # Auto-setup if requested
        if auto_setup:
            self.setup()
    
    def setup(self) -> None:
        """
        Setup the GEE_notebook_Forestry integration system.
        
        This method:
        1. Detects and resolves paths to GEE_notebook_Forestry
        2. Sets up Python path for imports
        3. Configures import strategy (local vs container)
        4. Validates basic dependencies
        """
        try:
            self.logger.info("Setting up GEE_notebook_Forestry integration...")
            
            # Use import manager to detect and setup GEE Forestry
            success, path, strategy = self.import_manager.detect_and_setup_gee_forestry()
            
            if success:
                self._gee_forestry_available = True
                self._gee_forestry_path = path
                self._import_strategy = strategy
                self.logger.info(f"GEE_notebook_Forestry integration setup: {path} (strategy: {strategy})")
            else:
                self._gee_forestry_available = False
                self.logger.warning("GEE_notebook_Forestry not found. Integration not available.")
            
            # Validate basic dependencies
            self.dependency_manager.validate_dependencies()
            
            self.logger.info("GEE_notebook_Forestry integration setup complete.")
            
        except Exception as e:
            self.logger.error(f"Setup failed: {e}")
            raise ForestryCarbonError(f"Failed to setup GEE_notebook_Forestry integration: {e}")
    
    
    def set_gee_forestry_path(self, path: Union[str, Path]) -> None:
        """
        Set the path to GEE_notebook_Forestry directory.
        
        Args:
            path: Path to GEE_notebook_Forestry directory
        """
        path = Path(path)
        if not path.exists():
            raise ForestryCarbonError(f"GEE_notebook_Forestry path does not exist: {path}")
        
        if not (path / "osi").exists():
            raise ForestryCarbonError(f"Invalid GEE_notebook_Forestry directory: {path}")
        
        self._gee_forestry_path = path
        self.logger.info(f"GEE_notebook_Forestry path set to: {path}")
    
    @property
    def gee_forestry_available(self) -> bool:
        """Check if GEE_notebook_Forestry is available."""
        return self._gee_forestry_available
    
    @property
    def gee_forestry_path(self) -> Optional[Path]:
        """Get GEE_notebook_Forestry path if available."""
        return self._gee_forestry_path
    
    @property
    def import_strategy(self) -> Optional[str]:
        """Get import strategy (local or container)."""
        return self._import_strategy
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get information about the current system setup.
        
        Returns:
            Dictionary with system information
        """
        return {
            'version': __version__,
            'gee_forestry_available': self._gee_forestry_available,
            'gee_forestry_path': str(self._gee_forestry_path) if self._gee_forestry_path else None,
            'import_strategy': self._import_strategy,
            'import_info': self.import_manager.get_import_info() if self._gee_forestry_available else None,
            'config': self.config
        }
    
    def get_import_guide(self) -> str:
        """
        Get import guide for GEE_notebook_Forestry.
        
        Returns:
            Formatted import guide
        """
        if self._gee_forestry_available:
            return self.import_manager.create_import_guide()
        else:
            return "GEE_notebook_Forestry not available. Please check setup requirements."
    
    def initialize_gee(self, project_id: Optional[str] = None) -> None:
        """
        Initialize Google Earth Engine using the forestry library's initialization pattern.
        
        This method follows the same initialization logic used in run_eligibility():
        1. Tries gee_lib auth first (container environment)
        2. Falls back to direct GEE initialization
        
        Args:
            project_id: Optional Google Cloud project ID. If None, uses GOOGLE_CLOUD_PROJECT env var.
        
        Raises:
            ForestryCarbonError: If GEE initialization fails
        """
        try:
            import ee
            from dotenv import load_dotenv
            
            # Load environment variables
            load_dotenv()
            
            # Check if already initialized
            try:
                # Try to get a simple property to check if initialized
                _ = ee.Number(1).getInfo()
                self.logger.info("GEE already initialized")
                return
            except Exception:
                # Not initialized, continue
                pass
            
            self.logger.info("Initializing Google Earth Engine...")
            
            # Try to import gee_lib auth if available
            try:
                # Try gee_lib auth (container environment)
                try:
                    from gee_lib.osi.auth import initialize_gee
                    project = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
                    if project:
                        initialize_gee(project_id=project, use_service_account=True)
                        self.logger.info("GEE initialized successfully using gee_lib auth")
                        return
                    else:
                        raise ValueError("GOOGLE_CLOUD_PROJECT not set")
                except ImportError:
                    # Fallback to direct GEE initialization
                    try:
                        if project_id:
                            ee.Initialize(project=project_id)
                        else:
                            project = os.getenv('GOOGLE_CLOUD_PROJECT')
                            if project:
                                ee.Initialize(project=project)
                            else:
                                ee.Initialize()
                        self.logger.info("GEE initialized directly")
                    except Exception as init_error:
                        self.logger.error(f"Failed to initialize GEE: {init_error}")
                        raise ForestryCarbonError(f"Failed to initialize GEE: {init_error}")
            except Exception as e:
                self.logger.warning(f"Could not use gee_lib auth: {e}. Attempting direct GEE initialization...")
                # Fallback to direct initialization
                try:
                    if project_id:
                        ee.Initialize(project=project_id)
                    else:
                        project = os.getenv('GOOGLE_CLOUD_PROJECT')
                        if project:
                            ee.Initialize(project=project)
                        else:
                            ee.Initialize()
                    self.logger.info("GEE initialized directly (fallback)")
                except Exception as init_error:
                    self.logger.error(f"Failed to initialize GEE: {init_error}")
                    raise ForestryCarbonError(f"Failed to initialize GEE: {init_error}")
        
        except ImportError:
            raise ForestryCarbonError("Earth Engine library not available. Install with: pip install earthengine-api geemap")
    
    def run_eligibility(self, 
                        config: Dict[str, Any],
                        use_gee: bool = True,
                        **kwargs) -> Dict[str, Any]:
        """
        Run the Forestry ARR Eligibility analysis workflow.
        
        This method implements the complete workflow from the notebook:
        1. Initialize GEE (if use_gee=True)
        2. Load AOI and training points
        3. Create image mosaic
        4. Calculate FCD (Forest Canopy Density)
        5. Calculate spectral indices and normalize
        6. OBIA segmentation
        7. ML classification using existing training points
        8. Hansen historical loss analysis
        9. Final zone assignment (FCD + ML + Hansen)
        
        Args:
            config: Configuration dictionary with all required parameters
                Required keys:
                - AOI_path: Path to AOI shapefile
                - input_training: Path to training points shapefile
                - OID_field_name: Field name for AOI ID
                - I_satellite: Satellite type ('Sentinel', 'Landsat', 'Planet')
                - date_start_end: [start_date, end_date]
                - project_name: Project name
                - cloud_cover_threshold: Cloud cover threshold
                - region: Region name
                - crs_input: Input CRS (e.g., 'EPSG:4326')
                - algo_ml_selected: ML algorithm ('rf', 'svm', 'gbm', 'cart')
                - fcd_selected: FCD method (11, 21, 12, 22)
                - high_forest: High forest threshold
                - yrf_forest: YRF forest threshold
                - shrub_grass: Shrub/grass threshold
                - open_land: Open land threshold
                - year_start_loss: Year start for loss analysis
                - tree_cover_forest: Tree cover threshold
                - pixel_number: Minimum pixel number
                - pca_scaling: PCA scaling factor
                - super_pixel_size: Super pixel size for OBIA
                - band_name_image: Band name for classification
                - ndwi_hi_sentinel: NDWI high threshold for Sentinel
                - ndwi_hi_landsat: NDWI high threshold for Landsat
                - ndwi_hi_planet: NDWI high threshold for Planet
            use_gee: Whether to use Google Earth Engine (True) or STAC (False)
                Currently only use_gee=True is implemented. use_gee=False returns None.
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
                - final_zone: Earth Engine Image with final zone classification
                - intermediate_results: Dictionary with intermediate processing results
                    - image_mosaick: Earth Engine Image of mosaicked satellite imagery
                    - FCD1_1, FCD2_1: Forest Canopy Density images
                    - training_points_info: Original validated training points (before ML split)
                        - training_points_ee: Earth Engine FeatureCollection
                        - num_points_before_validation: Count before validation
                        - num_points_after_validation: Count after filtering non-integer code_lu
                        - unique_classes: List of unique class codes
                    - ml_training_points: Actual training points used by ML classifier (after split/stratification)
                    - ml_validation_points: Actual validation points for confusion matrix (after split/stratification)
                    - classifier_results: Full ML classification results dictionary
                    - selected_image_lc: Selected land cover classification image
                    - algo_ml_selected: Selected ML algorithm name
                    - hansen_results: Hansen historical loss analysis results (contains treeLossYear, treeLoss, minLoss, gfc, etc.)
                    - treeLossYear: Direct access to tree loss year image
                    - treeLoss: Direct access to tree loss mask (years 0-23, masked)
                    - minLoss: Direct access to minimum loss image
                    - list_images_classified: FCD-based classification results
                    - fcd_classified_zone: FCD-based zone classification image (before ML overlay)
                    - zones_all_algorithms: Dictionary with zone classifications for all ML algorithms:
                        - 'rf': Zone classification using Random Forest land cover
                        - 'svm': Zone classification using SVM land cover
                        - 'gbm': Zone classification using GBM land cover
                        - 'cart': Zone classification using CART land cover
                    - HighForestDense: High forest density mask
                    - layer_names: Dictionary with standardized layer names for all outputs:
                        - image_mosaick: Layer name for mosaicked satellite image
                        - FCD1_1, FCD2_1: Layer names for Forest Canopy Density images
                        - treeLossYear, treeLoss, minLoss: Layer names for Hansen historical loss
                        - fcd_classified_zone: Layer name for FCD-based zone classification
                        - final_zone: Layer name for final zone classification (selected algorithm)
                        - zone_rf, zone_svm, zone_gbm, zone_cart: Layer names for zone classifications by ML algorithm
                        - land_cover_rf, land_cover_svm, land_cover_gbm, land_cover_cart: Layer names for ML classifiers
                        - land_cover_selected: Layer name for selected ML algorithm result
                - visualization_params: Dictionary with visualization parameters (direct access):
                    - mosaic: Visualization parameters for mosaicked image (dict)
                    - FCD1_1, FCD2_1: Visualization parameters for FCD layers (dict)
                    - land_cover: Visualization parameters for land cover classification (dict)
                    - zone: Visualization parameters for final zone classification (dict)
                    - land_cover_legend: Legend class mapping for land cover
                    - zone_legend: Legend class mapping for zones
                    - _metadata: Descriptions for each visualization (optional reference)
                - config: Updated configuration dictionary
                
        Raises:
            ForestryCarbonError: If GEE_notebook_Forestry is not available or processing fails
        """
        if not self._gee_forestry_available:
            raise ForestryCarbonError(
                "GEE_notebook_Forestry not available. Please ensure it is properly set up."
            )
        
        if not use_gee:
            # TODO: Implement STAC-based workflow
            self.logger.warning("use_gee=False (STAC) is not yet implemented. Returning None.")
            return None
        
        try:
            self.logger.info("Starting Forestry ARR Eligibility analysis workflow...")
            
            # Ensure config is a dict and has required parameters with defaults
            if not isinstance(config, dict):
                raise ForestryCarbonError("config must be a dictionary")
            
            # Set default values for optional parameters
            config.setdefault('IsThermal', False)
            
            # Import required modules dynamically
            import ee
            import geemap
            import geopandas as gpd
            
            # Import OSI modules based on import strategy
            import_strategy = self.import_strategy
            if import_strategy == 'container':
                # Container environment uses 'gee_lib' or 'osi' directly
                try:
                    from gee_lib.osi.image_collection.main import ImageCollection
                    from gee_lib.osi.spectral_indices.spectral_analysis import SpectralAnalysis
                    from gee_lib.osi.spectral_indices.utils import normalization_100
                    from gee_lib.osi.hansen.historical_loss import HansenHistorical
                    from gee_lib.osi.classifying.assign_zone import AssignClassZone
                    from gee_lib.osi.obia.main import OBIASegmentation
                    from gee_lib.osi.ml.main import LandcoverML
                    from gee_lib.osi.fcd.main_fcd import FCDCalc
                except ImportError:
                    # Fallback to direct osi import
                    try:
                        from osi.image_collection.main import ImageCollection
                        from osi.spectral_indices.spectral_analysis import SpectralAnalysis
                        from osi.spectral_indices.utils import normalization_100
                        from osi.hansen.historical_loss import HansenHistorical
                        from osi.classifying.assign_zone import AssignClassZone
                        from osi.obia.main import OBIASegmentation
                        from osi.ml.main import LandcoverML
                        from osi.fcd.main_fcd import FCDCalc
                    except ImportError as e:
                        raise ForestryCarbonError(f"Failed to import OSI modules: {e}")
            else:
                # Local development uses 'GEE_notebook_Forestry'
                try:
                    from GEE_notebook_Forestry.osi.image_collection.main import ImageCollection
                    from GEE_notebook_Forestry.osi.spectral_indices.spectral_analysis import SpectralAnalysis
                    from GEE_notebook_Forestry.osi.spectral_indices.utils import normalization_100
                    from GEE_notebook_Forestry.osi.hansen.historical_loss import HansenHistorical
                    from GEE_notebook_Forestry.osi.classifying.assign_zone import AssignClassZone
                    from GEE_notebook_Forestry.osi.obia.main import OBIASegmentation
                    from GEE_notebook_Forestry.osi.ml.main import LandcoverML
                    from GEE_notebook_Forestry.osi.fcd.main_fcd import FCDCalc
                except ImportError as e:
                    raise ForestryCarbonError(f"Failed to import OSI modules: {e}")
            
            # Step 1: Initialize GEE
            self.initialize_gee()
            
            # Step 2: Load and validate AOI
            self.logger.info("Loading AOI...")
            aoi_path = config['AOI_path']
            if not os.path.exists(aoi_path):
                raise ForestryCarbonError(f"AOI file not found: {aoi_path}")
            
            AOI = geemap.shp_to_ee(aoi_path)
            config['AOI'] = AOI
            
            # Validate AOI shapefile
            aoi_shp = gpd.GeoDataFrame.from_file(aoi_path)
            oid_field = config['OID_field_name']
            if oid_field not in aoi_shp.columns:
                raise ValueError(f"The field {oid_field} does not exist in the AOI shapefile.")
            
            if aoi_shp.crs != config['crs_input']:
                aoi_shp = aoi_shp.to_crs(config['crs_input'])
            
            # Create AOI image for overlaying
            AOI_img = AOI.filter(ee.Filter.notNull([oid_field])).reduceToImage(
                properties=[oid_field],
                reducer=ee.Reducer.first()
            )
            config['AOI_img'] = AOI_img
            
            # Step 3: Load and validate training points
            self.logger.info("Loading training points...")
            training_path = config['input_training']
            if not os.path.exists(training_path):
                raise ForestryCarbonError(f"Training points file not found: {training_path}")
            
            training_points_gdf = gpd.GeoDataFrame.from_file(training_path)
            
            # Check CRS
            if training_points_gdf.crs != config['crs_input']:
                training_points_gdf = training_points_gdf.to_crs(config['crs_input'])
            
            # Validate training points - filter non-integer code_lu values
            def is_integer(value):
                return isinstance(value, int)
            
            training_points_gdf['code_lu'] = training_points_gdf['code_lu'].apply(
                lambda x: x if is_integer(x) else None
            )
            
            training_points_ee = geemap.geopandas_to_ee(training_points_gdf)
            
            # Store validated training points info for reference
            training_points_info = {
                'training_points_ee': training_points_ee,  # Earth Engine FeatureCollection
                'num_points_before_validation': training_points_gdf.shape[0],
                'num_points_after_validation': training_points_gdf['code_lu'].notna().sum(),
                'unique_classes': sorted(training_points_gdf['code_lu'].dropna().unique().tolist()) if 'code_lu' in training_points_gdf.columns else []
            }
            
            # Step 4 & 5: Calculate FCD (FCDCalc creates ImageCollection internally and optimizes calls)
            # FCDCalc now computes image_mosaick only once in __init__, avoiding duplicate processing
            self.logger.info("Calculating Forest Canopy Density (FCD)...")
            # Ensure config has IsThermal before passing to FCDCalc
            if 'IsThermal' not in config:
                config['IsThermal'] = False
            
            # FCDCalc creates its own ImageCollection and computes image_mosaick once
            # This now only prints "selecting Sentinel images" once (from image_mosaick() in __init__)
            fcd_instance = FCDCalc(config)
            class_FCD_run = fcd_instance.fcd_calc()
            # Select the 'FCD' band explicitly for visualization (FCD images have single 'FCD' band)
            # FCD values can be negative or > 80, so clamp to 0-80 range for proper color visualization
            # Clip to AOI to ensure proper visualization boundaries
            FCD1_1 = class_FCD_run['FCD1_1'].select('FCD').clamp(0, 80).unmask(0).clip(AOI)
            FCD2_1 = class_FCD_run['FCD2_1'].select('FCD').clamp(0, 80).unmask(0).clip(AOI)
            
            # Reuse the image mosaic from FCDCalc (already computed in __init__)
            image_mosaick = fcd_instance.image_mosaick
            
            # Step 6: Calculate spectral indices
            self.logger.info("Calculating spectral indices...")
            classImageSpectral = SpectralAnalysis(image_mosaick, config)
            pca_scale = classImageSpectral.pca_scale
            
            ndwi_image = classImageSpectral.NDWI_func()
            msavi2_image = classImageSpectral.MSAVI2_func()
            mtvi2_image = classImageSpectral.MTVI2_func()
            ndvi_image = classImageSpectral.NDVI_func()
            vari_image = classImageSpectral.VARI_func()
            
            # Add spectral indices to image
            # Note: FCD1_1 and FCD2_1 are already selected with 'FCD' band
            image_mosaick_all_bands = image_mosaick.addBands([
                FCD2_1.rename('FCD2_1'),
                FCD1_1.rename('FCD1_1')
            ])
            
            image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari = (
                image_mosaick_all_bands
                .addBands(ndwi_image)
                .addBands(msavi2_image)
                .addBands(mtvi2_image)
                .addBands(ndvi_image)
                .addBands(vari_image)
            )
            
            # Step 7: Normalize images
            self.logger.info("Normalizing images...")
            red_norm = normalization_100(
                image_mosaick.select(['red']), 
                pca_scale=pca_scale, 
                AOI=AOI
            )
            green_norm = normalization_100(
                image_mosaick.select(['green']), 
                pca_scale=pca_scale, 
                AOI=AOI
            )
            blue_norm = normalization_100(
                image_mosaick.select(['blue']), 
                pca_scale=pca_scale, 
                AOI=AOI
            )
            nir_norm = normalization_100(
                image_mosaick.select(['nir']), 
                pca_scale=pca_scale, 
                AOI=AOI
            )
            
            image_norm = red_norm.addBands(green_norm).addBands(blue_norm).addBands(nir_norm)
            
            # Normalize spectral indices
            image_norm_ndvi = normalization_100(
                image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari.select('NDVI'),
                pca_scale=pca_scale,
                AOI=AOI
            )
            image_norm_ndwi = normalization_100(
                image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari.select('ndwi'),
                pca_scale=pca_scale,
                AOI=AOI
            )
            image_norm_msavi2 = normalization_100(
                image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari.select('msavi2'),
                pca_scale=pca_scale,
                AOI=AOI
            )
            image_norm_mtvi2 = normalization_100(
                image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari.select('MTVI2'),
                pca_scale=pca_scale,
                AOI=AOI
            )
            image_norm_vari = normalization_100(
                image_mosaick_ndvi_ndwi_msavi2_mtvi2_vari.select('VARI'),
                pca_scale=pca_scale,
                AOI=AOI
            )
            
            image_norm_with_spectral_indices = (
                image_norm
                .addBands(image_norm_ndvi)
                .addBands(image_norm_ndwi)
                .addBands(image_norm_msavi2)
                .addBands(image_norm_mtvi2)
                .addBands(image_norm_vari)
            )
            
            # Add FCD bands to normalized image (FCD1_1 and FCD2_1 are already selected with 'FCD' band)
            image_norm_with_spectral_indices_FCD = (
                image_norm_with_spectral_indices
                .addBands(FCD2_1.rename('FCD2_1'))
                .addBands(FCD1_1.rename('FCD1_1'))
            )
            
            # Step 8: OBIA Segmentation
            self.logger.info("Performing OBIA segmentation...")
            obia = OBIASegmentation(
                config=config,
                image=image_norm_with_spectral_indices_FCD,
                pca_scale=pca_scale
            )
            
            clusters = obia.SNIC_cluster()['clusters']
            object_properties_image = obia.summarize_cluster(is_include_std=False)
            object_properties_image = object_properties_image.clip(AOI).toFloat()
            
            # Step 9: Add date analyzed to config
            config["date_analyzed"] = datetime.now().strftime("%Y-%m-%d")
            
            # Step 10: ML Classification using existing training points
            self.logger.info("Running ML classification...")
            lc = LandcoverML(
                config=config,
                input_image=image_norm_with_spectral_indices_FCD,
                cluster_properties=object_properties_image,
                pca_scale=pca_scale
            )
            
            classifier = lc.run_classifier()
            
            # Extract actual training and validation points used for ML (after split/stratification)
            # These are the points used for training the classifier and generating confusion matrix
            ml_training_points = classifier['training_points']  # Actual training points used by ML
            ml_validation_points = classifier['validation_points']  # Actual validation points for confusion matrix
            
            # Get land cover legend parameters (visualization params for land cover)
            legend_lc = lc.lc_legend_param()
            vis_param_lc = legend_lc['vis_param_lc']
            legend_class_lc = legend_lc['legend_class']
            
            # Get selected ML algorithm result
            algo_ml_selected = config['algo_ml_selected']
            if algo_ml_selected == 'rf':
                selected_image_lc = classifier['classified_image_rf']
            elif algo_ml_selected == 'svm':
                selected_image_lc = classifier['classified_image_svm']
            elif algo_ml_selected == 'gbm':
                selected_image_lc = classifier['classified_image_gbm']
            elif algo_ml_selected == 'cart':
                selected_image_lc = classifier['classified_image_cart']
            else:
                self.logger.warning(f"Unknown algorithm {algo_ml_selected}, defaulting to rf")
                selected_image_lc = classifier['classified_image_rf']
                algo_ml_selected = 'rf'
            
            # Step 11: Hansen Historical Loss Analysis
            self.logger.info("Analyzing Hansen historical tree loss...")
            hansen_class = HansenHistorical(config)
            run_hansen = hansen_class.initiate_tcl()
            
            treeLossYear = run_hansen['treeLossYear']
            # Create treeLoss mask: tree loss between years 0-23 (2000-2023)
            treeLoss = treeLossYear.gte(0).And(treeLossYear.lte(23)).selfMask()
            minLoss = run_hansen['minLoss']
            gfc = run_hansen['gfc']
            
            # Add treeLoss to hansen results for consistency
            run_hansen['treeLoss'] = treeLoss
            
            # Step 12: Assign FCD zones
            self.logger.info("Assigning FCD classification zones...")
            class_assigning_fcd = AssignClassZone(
                config=config,
                FCD1_1=FCD1_1,
                FCD2_1=FCD2_1
            )
            
            list_images_classified = class_assigning_fcd.assigning_fcd_class(gfc, minLoss)
            HighForestDense = list_images_classified['HighForestDense']
            fcd_classified_zone = list_images_classified['all_zone']  # FCD-based zone classification (before ML overlay)
            
            # Step 13: Final zone assignment (ML + FCD + Hansen) for all ML algorithms
            self.logger.info("Computing final zone classification for all ML algorithms...")
            
            # Generate zone classifications for all ML algorithms
            zones_all_algorithms = {}
            zones_all_algorithms['rf'] = class_assigning_fcd.assign_zone_ml(
                classifier['classified_image_rf'],
                minLoss,
                AOI_img,
                HighForestDense
            )
            zones_all_algorithms['svm'] = class_assigning_fcd.assign_zone_ml(
                classifier['classified_image_svm'],
                minLoss,
                AOI_img,
                HighForestDense
            )
            zones_all_algorithms['gbm'] = class_assigning_fcd.assign_zone_ml(
                classifier['classified_image_gbm'],
                minLoss,
                AOI_img,
                HighForestDense
            )
            zones_all_algorithms['cart'] = class_assigning_fcd.assign_zone_ml(
                classifier['classified_image_cart'],
                minLoss,
                AOI_img,
                HighForestDense
            )
            
            # Get the selected algorithm's zone
            final_zone = zones_all_algorithms[algo_ml_selected]
            
            # Get visualization parameters for zones (from AssignClassZone)
            vis_param_zone = class_assigning_fcd.vis_param_merged
            legend_class_zone = class_assigning_fcd.legend_class
            
            # Extract variables for layer name generation
            I_satellite = config['I_satellite']
            project_name = config['project_name']
            start_date = config['date_start_end'][0]
            end_date = config['date_start_end'][1]
            
            # Create visualization parameters for mosaic image based on satellite type
            if I_satellite == 'Planet':
                vis_param_mosaic = {"bands": ["red", "green", "blue"], "min": 0, "max": 0.6, "gamma": 1.5}
            else:
                # Sentinel or Landsat
                vis_param_mosaic = {'bands': ['swir2', 'nir', 'red'], 'min': 0, 'max': 0.6, 'gamma': 1.5}
            
            # FCD visualization parameters (standard for FCD layers)
            # Note: FCD images are single-band with 'FCD' band name, already clamped to 0-80 range
            # Palette: red (low density) -> yellow (medium) -> green (high density)
            fcd_visparams = {'min': 0, 'max': 80, 'palette': ['ff4c16', 'ffd96c', '39a71d']}
            
            # Prepare visualization parameters dictionary
            # Flattened structure for easier access: result['visualization_params']['FCD1_1'] directly returns vis_param
            visualization_params = {
                'mosaic': vis_param_mosaic,
                'FCD1_1': fcd_visparams,
                'FCD2_1': fcd_visparams,
                'land_cover': vis_param_lc,
                'zone': vis_param_zone,
                # Metadata for reference
                '_metadata': {
                    'mosaic': f'{I_satellite} mosaicked image visualization',
                    'FCD1_1': 'Forest Canopy Density 1 visualization',
                    'FCD2_1': 'Forest Canopy Density 2 visualization',
                    'land_cover': 'Land cover classification visualization',
                    'zone': 'Final zone classification visualization'
                },
                # Legend classes for land cover and zone
                'land_cover_legend': legend_class_lc,
                'zone_legend': legend_class_zone
            }
            
            # Generate standardized layer names
            layer_names = {
                'image_mosaick': f'{I_satellite} mosaicked - {start_date}-{end_date} VegColor',
                'FCD1_1': f'FCD1_1_{project_name}',
                'FCD2_1': f'FCD2_1_{project_name}',
                'treeLossYear': 'treeLossYear',
                'treeLoss': 'treeLoss',
                'minLoss': 'minLoss',
                'fcd_classified_zone': f'FCD_classified_zone_{project_name}',
                'final_zone': f'Final_zone_ML_{algo_ml_selected}_Hansen',
                'zone_rf': f'Final_zone_ML_rf_Hansen',
                'zone_svm': f'Final_zone_ML_svm_Hansen',
                'zone_gbm': f'Final_zone_ML_gbm_Hansen',
                'zone_cart': f'Final_zone_ML_cart_Hansen',
                'land_cover_rf': 'Random_forest_lc_result',
                'land_cover_svm': 'SVM_lc_result',
                'land_cover_gbm': 'GBM_lc_result',
                'land_cover_cart': 'CART_lc_result',
                'land_cover_selected': f'{algo_ml_selected.upper()}_lc_result'
            }
            
            # Prepare results
            intermediate_results = {
                'image_mosaick': image_mosaick,
                'FCD1_1': FCD1_1,
                'FCD2_1': FCD2_1,
                'image_norm_with_spectral_indices_FCD': image_norm_with_spectral_indices_FCD,
                'training_points_info': training_points_info,  # Original validated training points (before ML split)
                'ml_training_points': ml_training_points,  # Actual training points used by ML classifier (after split)
                'ml_validation_points': ml_validation_points,  # Actual validation points for confusion matrix (after split)
                'classifier_results': classifier,  # Full ML classifier results
                'selected_image_lc': selected_image_lc,
                'algo_ml_selected': algo_ml_selected,
                'hansen_results': run_hansen,  # Contains treeLossYear, treeLoss, minLoss, gfc, etc.
                'treeLossYear': treeLossYear,  # Direct access to treeLossYear
                'treeLoss': treeLoss,  # Direct access to treeLoss (masked tree loss 0-23 years)
                'minLoss': minLoss,  # Direct access to minLoss
                'list_images_classified': list_images_classified,  # FCD-based classification results
                'fcd_classified_zone': fcd_classified_zone,  # FCD-based zone classification (before ML overlay)
                'zones_all_algorithms': zones_all_algorithms,  # Zone classifications for all ML algorithms (rf, svm, gbm, cart)
                'HighForestDense': HighForestDense,
                'layer_names': layer_names  # Standardized layer names for visualization/export
            }
            
            self.logger.info("Forestry ARR Eligibility analysis workflow completed successfully!")
            
            return {
                'final_zone': final_zone,
                'intermediate_results': intermediate_results,
                'visualization_params': visualization_params,
                'config': config
            }
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            raise ForestryCarbonError(f"Forestry ARR Eligibility analysis failed: {e}")
    
    def get_ds_resampled_gee(self,
                             config: Optional[Dict[str, Any]] = None,
                             zarr_path: Optional[str] = None,
                             use_existing_asset: bool = False,
                             asset_folder: Optional[str] = None,
                             asset_is_monthly_composites: bool = False,
                             use_processed_asset: bool = False,
                             processed_asset_path: Optional[str] = None,
                             years_back: int = 10,
                             pixel_scale: Optional[int] = None,
                             chunk_sizes: Optional[Dict[str, int]] = None,
                             compression: str = 'lz4',
                             compression_level: int = 1,
                             overwrite_zarr: bool = False,
                             save_to_zarr: bool = True,
                             storage: str = 'auto',
                             **kwargs) -> 'xr.Dataset':
        """
        Process GEE ImageCollection to get resampled xarray Dataset (ds_resampled).
        
        This method implements the complete workflow for processing GEE data into xarray
        format ready for time series analysis (tsfresh). It:
        1. Creates/prepares GEE ImageCollection with UTM reprojection
        2. Filters by cloud cover
        3. Creates monthly composites
        4. Adds spectral indices (NDVI, EVI) and applies smoothing
        5. Converts to xarray Dataset using xee engine
        6. Saves/loads from zarr for efficient storage
        7. Returns ds_resampled ready for tsfresh processing
        
        Args:
            config: Configuration dictionary. If None, uses self.config.
                Required keys:
                - AOI_path: Path to AOI shapefile
                - I_satellite: Satellite type ('Sentinel', 'Landsat', 'Planet')
                - date_start_end: [start_date, end_date]
                - cloud_cover_threshold: Cloud cover threshold
                - region: Region name
                - crs_input: Input CRS (e.g., 'EPSG:4326')
            zarr_path: Path to zarr store (local or GCS URI like gs://bucket/path.zarr).
                If None, uses GCS_ZARR_DIR env var + '/ds_resampled.zarr'
            use_existing_asset: If True, use existing GEE asset instead of creating new collection.
                Default False.
            asset_folder: GEE asset folder path (e.g., 'projects/xxx/assets/yyy').
                Required if use_existing_asset=True.
            asset_is_monthly_composites: If True (and use_existing_asset=True), the asset contains
                monthly composites, so skip cloud filtering and compositing steps. Default False.
            use_processed_asset: If True, load the final processed ImageCollection (with indices and smoothing)
                directly from a GEE asset, skipping all processing steps. Default False.
            processed_asset_path: GEE asset path to the processed ImageCollection (e.g., 
                'projects/xxx/assets/yyy/processed_collection'). Required if use_processed_asset=True.
            years_back: Number of years to look back from end_date for historical data.
                Default 10.
            pixel_scale: Pixel scale in meters. If None, auto-detects based on satellite type.
                Sentinel-2: 10m, Landsat: 30m
            chunk_sizes: Chunk sizes for zarr storage. Default: {'time': 40, 'x': 1024, 'y': 1024}
            compression: Compression algorithm ('lz4', 'blosc', 'zstd', or None). Default 'lz4'.
            compression_level: Compression level (1-9). Default 1 (fastest).
            overwrite_zarr: Whether to overwrite existing zarr store. Default False.
            save_to_zarr: Whether to save dataset to zarr (GCS or local). If False, returns dataset directly
                from GEE without saving. Default True. Set to False to avoid GCS zarr export.
            storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
            **kwargs: Additional keyword arguments passed to xr.open_dataset.
            
        Returns:
            xarray.Dataset: The converted dataset ready for further processing
            
        Raises:
            ForestryCarbonError: If GEE_notebook_Forestry is not available or processing fails
            ImportError: If required libraries (xarray, xee, gcsfs) are not available
        """
        try:
            import xarray as xr
        except ImportError:
            raise ImportError("xarray is required. Install with: pip install xarray")
        
        try:
            import xee
        except ImportError:
            raise ImportError("xee is required. Install with: pip install xee")
        
        if not self._gee_forestry_available:
            raise ForestryCarbonError(
                "GEE_notebook_Forestry not available. Please ensure it is properly set up."
            )
        
        # Use provided config or self.config
        if config is None:
            config = self.config
        
        if not isinstance(config, dict):
            raise ForestryCarbonError("config must be a dictionary")
        
        # Import required modules
        import ee
        import geemap
        import geopandas as gpd
        import os
        from ..utils.zarr_utils import save_dataset_efficient_zarr, load_dataset_zarr
        from ..utils.gee_processing import (
            prepare_image_collection_for_processing,
            filter_by_cloud_cover,
            create_monthly_composites,
            rename_composite_bands,
            process_collection_with_indices_and_smoothing
        )
        from .utils import DataUtils
        
        self.logger.info("Starting GEE to xarray conversion workflow (get_ds_resampled_gee)...")
        
        # Step 1: Load AOI
        self.logger.info("Loading AOI...")
        data_utils = DataUtils(config, use_gee=True)
        aoi_gpd, aoi_ee = data_utils.load_geodataframe_gee(config['AOI_path'])
        
        # Check if we should load processed collection directly from GEE asset
        if use_processed_asset:
            if processed_asset_path is None:
                raise ValueError("processed_asset_path is required when use_processed_asset=True")
            
            self.logger.info(f"Loading processed ImageCollection from GEE asset: {processed_asset_path}")
            collection_with_sg = ee.ImageCollection(processed_asset_path)
            
            # Get UTM CRS and pixel scale from config or auto-detect
            if 'utm_crs' in config:
                utm_crs = config['utm_crs']
            else:
                # Auto-detect UTM CRS
                from ..utils.gee_processing import calculate_utm_crs
                utm_crs, utm_epsg, hemisphere = calculate_utm_crs(aoi_gpd)
                config['utm_crs'] = utm_crs
            
            if pixel_scale is None:
                from ..utils.gee_processing import get_pixel_scale
                pixel_scale = get_pixel_scale(config.get('I_satellite', 'Sentinel'))
            
            self.logger.info(f"Using processed asset. UTM CRS: {utm_crs}, Pixel scale: {pixel_scale}m")
        else:
            # Check if asset contains monthly composites
            if use_existing_asset and asset_is_monthly_composites:
                self.logger.info(f"Loading monthly composites from GEE asset: {asset_folder}")
                self.logger.info(f"  use_existing_asset={use_existing_asset}, asset_is_monthly_composites={asset_is_monthly_composites}")
                
                # Load asset directly as monthly composites
                collection_monthly_raw = ee.ImageCollection(asset_folder)
                
                # IMPORTANT: Rename bands immediately (remove _median suffix, exclude cloudM)
                # This is needed because the asset has bands like blue_median, but processing expects blue
                self.logger.info("Renaming bands (removing _median suffix)...")
                collection_monthly = rename_composite_bands(
                    collection_monthly_raw,
                    remove_suffix='_median',
                    exclude_bands=['cloudM']
                )
                
                # Get UTM CRS and pixel scale from config or auto-detect
                if 'utm_crs' in config:
                    utm_crs = config['utm_crs']
                else:
                    from ..utils.gee_processing import calculate_utm_crs
                    utm_crs, utm_epsg, hemisphere = calculate_utm_crs(aoi_gpd)
                    config['utm_crs'] = utm_crs
                
                if pixel_scale is None:
                    from ..utils.gee_processing import get_pixel_scale
                    pixel_scale = get_pixel_scale(config.get('I_satellite', 'Sentinel'))
                
                self.logger.info(f"✅ Loaded {collection_monthly.size().getInfo()} monthly composite images")
                self.logger.info(f"UTM CRS: {utm_crs}, Pixel scale: {pixel_scale}m")
                
            else:
                # Step 2: Create ImageCollection and prepare (UTM + reproject)
                self.logger.info("Creating ImageCollection and preparing for processing...")
                raw_collection, utm_crs, pixel_scale_auto, utm_epsg = prepare_image_collection_for_processing(
                    config=config,
                    aoi_gpd=aoi_gpd,
                    aoi_ee=aoi_ee,
                    years_back=years_back,
                    use_existing_asset=use_existing_asset,
                    asset_folder=asset_folder,
                    import_strategy=self.import_strategy,
                    reproject_to_utm_flag=True
                )
                
                # Use provided pixel_scale or auto-detected
                if pixel_scale is None:
                    pixel_scale = pixel_scale_auto
                
                config['utm_crs'] = utm_crs
                self.logger.info(f"UTM CRS: {utm_crs}, Pixel scale: {pixel_scale}m")
                
                # Step 3: Filter by cloud cover
                self.logger.info("Filtering by cloud cover...")
                aoi_gpd_utm = aoi_gpd.to_crs(utm_crs)
                aoi_ee_utm_geom = geemap.geopandas_to_ee(aoi_gpd_utm).geometry()
                aoi_bounds_utm = aoi_ee_utm_geom.bounds(maxError=1)
                
                valid_pixel_threshold = config.get('valid_pixel_threshold', 70.0)
                collection_filtered, stats = filter_by_cloud_cover(
                    raw_collection,
                    aoi_bounds_utm,
                    scale=pixel_scale,
                    crs=utm_crs,
                    valid_pixel_threshold=valid_pixel_threshold
                )
                
                # Step 4: Create monthly composites and rename bands
                self.logger.info("Creating monthly composites...")
                monthly_images = create_monthly_composites(
                    collection_filtered,
                    aoi_ee.geometry(),
                    reducer='median'
                )
                
                collection_monthly_raw = ee.ImageCollection(monthly_images).sort('system:time_start')
                collection_monthly = rename_composite_bands(
                    collection_monthly_raw,
                    remove_suffix='_median',
                    exclude_bands=['cloudM']
                )
            
            # Step 5: Add spectral indices and apply smoothing
            self.logger.info("Adding spectral indices and applying smoothing...")
            collection_with_sg = process_collection_with_indices_and_smoothing(
                collection=collection_monthly,
                config=config,
                aoi_ee=aoi_ee,
                spectral_bands=['NDVI', 'EVI'],
                smoothing_window=3,
                smoothing_polyorder=2,
                add_fcd=False
            )
        
        # Prepare AOI geometry for xee conversion
        aoi_gpd_utm = aoi_gpd.to_crs(utm_crs)
        
        # Step 6: Convert to xarray using xee
        self.logger.info("Converting GEE ImageCollection to xarray using xee...")
        aoi_ee_utm_geom = geemap.geopandas_to_ee(aoi_gpd_utm).geometry()
        ds = xr.open_dataset(
            collection_with_sg,
            engine='ee',
            crs=utm_crs,
            scale=pixel_scale,
            geometry=aoi_ee_utm_geom,
            **kwargs
        )
        
        self.logger.info(f"Dataset created: {dict(ds.dims)}")
        
        # Step 7: Save to zarr or return directly
        if not save_to_zarr:
            self.logger.info("Skipping zarr save - returning dataset directly from GEE")
            self.logger.warning("⚠️  Dataset is still lazy-loaded from GEE. Accessing data may cause 'Too many concurrent aggregations' errors.")
            return ds
        
        # Step 8: Determine zarr path
        if zarr_path is None:
            zarr_path = os.getenv('GCS_ZARR_DIR', '')
            if not zarr_path:
                # Fallback to local path
                zarr_path = os.path.join(os.getcwd(), 'data', 'ds_resampled.zarr')
            else:
                zarr_path = os.path.join(zarr_path, 'ds_resampled.zarr')
        
        # Step 9: Save or load from zarr
        if storage == 'auto':
            storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
        
        # Check if zarr exists
        zarr_exists = False
        if storage == 'gcs':
            try:
                import gcsfs
                fs = gcsfs.GCSFileSystem(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
                zarr_exists = fs.exists(zarr_path)
            except Exception:
                zarr_exists = False
        else:
            zarr_exists = os.path.exists(zarr_path)
        
        if zarr_exists and not overwrite_zarr:
            self.logger.info(f"Loading existing zarr dataset from: {zarr_path}")
            ds_raw = load_dataset_zarr(zarr_path, storage=storage)
        else:
            self.logger.info("Preparing dataset for zarr storage...")
            ds_raw = ds
            
            # Save to zarr
            if chunk_sizes is None:
                chunk_sizes = {'time': 40, 'x': 1024, 'y': 1024}
            
            self.logger.info(f"Saving dataset to zarr: {zarr_path}")
            save_dataset_efficient_zarr(
                ds_raw,
                zarr_path,
                chunk_sizes=chunk_sizes,
                compression=compression,
                compression_level=compression_level,
                overwrite=overwrite_zarr,
                storage=storage
            )
            
            # Reload to ensure consistency
            ds_raw = load_dataset_zarr(zarr_path, storage=storage)
        
        self.logger.info("GEE to xarray conversion workflow completed successfully!")
        return ds_raw
    
    def prepare_tsfresh_with_ground_truth(
        self,
        ds_resampled: Optional['xr.Dataset'] = None,
        ground_truth_path: Optional[str] = None,
        ground_truth_gdf: Optional[Any] = None,
        buffer_pixels: int = 50,
        chunk_sizes: Optional[Dict[str, int]] = None,
        save_to_zarr: bool = True,
        zarr_path: Optional[str] = None,
        overwrite_zarr: bool = False,
        storage: str = 'auto'
    ) -> List['xr.Dataset']:
        """
        Prepare satellite data with ground truth labels for tsfresh feature extraction.
        
        This method:
        1. Loads ground truth training data (polygons)
        2. Clips satellite data to sample bounding boxes
        3. Converts training polygons to raster masks
        4. Merges masks into 4D dataset (plot_id, time, y, x)
        5. Merges satellite data with ground truth labels
        6. Optionally saves to zarr
        
        Args:
            ds_resampled: Satellite dataset from get_ds_resampled_gee(). If None, will call
                get_ds_resampled_gee() automatically.
            ground_truth_path: Path to ground truth parquet file (GCS or local).
                Required if ground_truth_gdf is None.
            ground_truth_gdf: Pre-loaded GeoDataFrame with ground truth data.
                Required if ground_truth_path is None.
            buffer_pixels: Buffer size in pixels around sample bounding boxes. Default 50.
            chunk_sizes: Chunk sizes for output dataset. Default: {'plot_id': 1, 'time': 20, 'x': 128, 'y': 128}
            save_to_zarr: Whether to save datasets to zarr. Default True.
            zarr_path: Base path for zarr stores. If None, uses GCS_ZARR_DIR env var.
            overwrite_zarr: Whether to overwrite existing zarr stores. Default False.
            storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
        
        Returns:
            List of xarray.Dataset, one per sample (layer), each with:
            - Dimensions: (plot_id, time, x, y)
            - Variables: [EVI, NDVI, ground_truth, gt_valid]
        """
        import geopandas as gpd
        from ..utils.tsfresh_utils import (
            load_ground_truth_data,
            prepare_tsfresh_data_with_ground_truth
        )
        from ..utils.zarr_utils import save_dataset_efficient_zarr, load_dataset_zarr
        import os
        
        self.logger.info("=" * 60)
        self.logger.info("Preparing tsfresh data with ground truth")
        self.logger.info("=" * 60)
        
        # Step 1: Get ds_resampled if not provided
        if ds_resampled is None:
            self.logger.info("ds_resampled not provided, calling get_ds_resampled_gee()...")
            ds_resampled = self.get_ds_resampled_gee(save_to_zarr=False)
        
        # Step 2: Load ground truth data
        if ground_truth_gdf is None:
            if ground_truth_path is None:
                raise ValueError("Either ground_truth_path or ground_truth_gdf must be provided")
            
            # Determine if GCS or local
            if ground_truth_path.startswith('gs://'):
                training_gdf = load_ground_truth_data(gcs_path=ground_truth_path)
            else:
                training_gdf = load_ground_truth_data(local_path=ground_truth_path)
        else:
            training_gdf = ground_truth_gdf
            self.logger.info(f"Using provided ground truth GeoDataFrame: {len(training_gdf)} polygons")
        
        # Step 3: Prepare tsfresh data with ground truth
        self.logger.info("Preparing tsfresh data with ground truth labels...")
        ds_gt_list = prepare_tsfresh_data_with_ground_truth(
            ds_resampled=ds_resampled,
            training_gdf=training_gdf,
            buffer_pixels=buffer_pixels,
            chunk_sizes=chunk_sizes
        )
        
        # Step 4: Save to zarr if requested
        if save_to_zarr:
            self.logger.info("Saving datasets to zarr...")
            
            # Determine zarr path
            if zarr_path is None:
                zarr_path = os.getenv('GCS_ZARR_DIR', '')
                if not zarr_path:
                    zarr_path = os.path.join(os.getcwd(), 'data', 'tsfresh_gt')
                else:
                    if not zarr_path.startswith('gs://'):
                        zarr_path = f"gs://{zarr_path}/tsfresh_gt"
                    else:
                        zarr_path = f"{zarr_path}/tsfresh_gt"
            
            # Determine storage
            if storage == 'auto':
                storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
            
            # Save each sample dataset
            saved_datasets = []
            for i, ds_gt in enumerate(ds_gt_list):
                plot_id = ds_gt.coords['plot_id'].values[0] if 'plot_id' in ds_gt.coords else f'sample_{i+1}'
                sample_zarr_path = f"{zarr_path}/{plot_id}.zarr"
                
                self.logger.info(f"Saving {plot_id} to: {sample_zarr_path}")
                
                # Check if exists
                zarr_exists = False
                if storage == 'gcs':
                    try:
                        import gcsfs
                        fs = gcsfs.GCSFileSystem(project=os.getenv('GOOGLE_CLOUD_PROJECT'))
                        zarr_exists = fs.exists(sample_zarr_path)
                    except Exception:
                        zarr_exists = False
                else:
                    zarr_exists = os.path.exists(sample_zarr_path)
                
                if zarr_exists and not overwrite_zarr:
                    self.logger.info(f"Loading existing zarr: {sample_zarr_path}")
                    ds_gt_loaded = load_dataset_zarr(sample_zarr_path, storage=storage)
                    saved_datasets.append(ds_gt_loaded)
                else:
                    # Save to zarr
                    if chunk_sizes is None:
                        chunk_sizes = {'plot_id': 1, 'time': 20, 'x': 128, 'y': 128}
                    
                    save_dataset_efficient_zarr(
                        ds_gt,
                        sample_zarr_path,
                        chunk_sizes=chunk_sizes,
                        compression='lz4',
                        compression_level=1,
                        overwrite=overwrite_zarr,
                        storage=storage
                    )
                    
                    # Reload to ensure consistency
                    ds_gt_loaded = load_dataset_zarr(sample_zarr_path, storage=storage)
                    saved_datasets.append(ds_gt_loaded)
            
            self.logger.info(f"✅ Saved {len(saved_datasets)} datasets to zarr")
            return saved_datasets
        else:
            self.logger.info("Skipping zarr save - returning datasets directly")
            return ds_gt_list
    
    def _apply_temporal_smoothing(self,
                                  ds: 'xr.Dataset',
                                  window_length: int = 3,
                                  polyorder: int = 2,
                                  resample_freq: Optional[str] = None) -> 'xr.Dataset':
        """
        Apply Savitzky-Golay temporal smoothing to xarray dataset.
        
        Args:
            ds: Input xarray Dataset
            window_length: Window length for smoothing (must be odd)
            polyorder: Polynomial order for smoothing
            resample_freq: Optional resampling frequency (e.g., 'MS' for monthly)
            
        Returns:
            Smoothed xarray Dataset
        """
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            raise ImportError("scipy is required for smoothing. Install with: pip install scipy")
        
        import pandas as pd
        import numpy as np
        import xarray as xr
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        if window_length < polyorder + 1:
            window_length = polyorder + 1
            if window_length % 2 == 0:
                window_length += 1
        
        self.logger.info(f"Applying Savitzky-Golay smoothing (window={window_length}, polyorder={polyorder})...")
        
        # Apply smoothing to each data variable
        smoothed_data = {}
        for var_name in ds.data_vars:
            var_data = ds[var_name]
            
            # Chunk for efficient processing
            var_chunked = var_data.chunk({"time": -1, "y": 256, "x": 256})
            
            # Apply smoothing function
            def smooth_1d(ts):
                if np.all(np.isnan(ts)):
                    return ts
                try:
                    return savgol_filter(ts, window_length=window_length, polyorder=polyorder, mode="interp")
                except Exception:
                    return ts
            
            smoothed_var = xr.apply_ufunc(
                smooth_1d,
                var_chunked,
                input_core_dims=[["time"]],
                output_core_dims=[["time"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[var_data.dtype],
            )
            
            smoothed_data[var_name] = smoothed_var
        
        # Create smoothed dataset
        ds_smoothed = xr.Dataset(smoothed_data, coords=ds.coords, attrs=ds.attrs)
        
        # Apply resampling if requested
        if resample_freq:
            self.logger.info(f"Resampling to {resample_freq}...")
            ds_smoothed = ds_smoothed.resample(time=resample_freq).mean()
            ds_smoothed = ds_smoothed.compute()
        
        return ds_smoothed
    
    def get_ds_sampled_mpc(
        self,
        ds_resampled: Optional['xr.Dataset'] = None,
        zarr_path: Optional[str] = None,
        years_back: int = 11,
        days_offset: int = 15,
        aoi_size_threshold_ha: float = 5000.0,
        max_examples: int = 5,
        max_retries: int = 3,
        create_visualization: bool = True,
        save_map: bool = False,
        map_filename: Optional[str] = None,
        storage: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Get sampled MPC (Microsoft Planetary Computer) data by analyzing duplicate STAC scenes.
        
        This method:
        1. Loads or accepts ds_resampled dataset
        2. Calculates MPC date range based on dataset's minimum date
        3. Determines AOI size and approach (GEE vs STAC)
        4. Searches for duplicate STAC scenes with different processing versions
        5. Optionally creates visualization map
        6. Returns analysis results
        
        Args:
            ds_resampled: Pre-loaded xarray Dataset. If None, will load from zarr_path.
            zarr_path: Path to zarr store containing ds_resampled. Required if ds_resampled is None.
                Can be GCS path (gs://...) or local path.
            years_back: Number of years to go back from end year for MPC date range. Default 11.
            days_offset: Days to subtract from min_date for end date. Default 15.
            aoi_size_threshold_ha: AOI size threshold in hectares to determine approach.
                If AOI > threshold, uses GEE approach. Default 5000.0 ha.
            max_examples: Maximum number of duplicate examples to show. Default 5.
            max_retries: Maximum number of retry attempts for STAC search. Default 3.
            create_visualization: Whether to create folium visualization map. Default True.
            save_map: Whether to save the map as HTML file. Default False.
            map_filename: Custom filename for the saved map. If None, auto-generates.
            storage: Storage type for zarr ('auto', 'local', 'gcs'). Default 'auto'.
        
        Returns:
            Dictionary with keys:
                - 'use_gee': bool - Whether to use GEE approach based on AOI size
                - 'aoi_ha': float - AOI area in hectares
                - 'mpc_date_range': List[str] - [start_date, end_date] for MPC search
                - 'selected_scene_data': Optional[Dict] - Selected duplicate scene data
                - 'visualization': Optional[folium.Map] - Visualization map if created (display directly or save to HTML)
                - 'bbox': shapely.geometry.box - Bounding box used for search
                - 'aoi_gpd': GeoDataFrame - Area of interest GeoDataFrame
                - 'aoi_ee': ee.Geometry - Earth Engine geometry for AOI
                - 'ds_resampled': xr.Dataset - Resampled dataset
                - 'mpc_urls': Dict[str, str] - Microsoft Planetary Computer Explorer URLs
        """
        import geopandas as gpd
        from shapely.geometry import box
        from ..utils.stac_utils import (
            search_and_analyze_duplicate_stac_scenes,
            create_unified_stac_visualization,
            calculate_mpc_date_range
        )
        from ..utils.zarr_utils import load_dataset_zarr
        from .utils import DataUtils
        import os
        
        self.logger.info("=" * 60)
        self.logger.info("MPC Eligibility Analysis")
        self.logger.info("=" * 60)
        
        # Step 1: Load ds_resampled if not provided
        if ds_resampled is None:
            if zarr_path is None:
                # Try to auto-detect from environment
                zarr_path = os.getenv('GCS_ZARR_DIR', '')
                if not zarr_path:
                    raise ValueError("Either ds_resampled or zarr_path must be provided")
                else:
                    zarr_path = os.path.join(zarr_path, 'ds_resampled.zarr')
            
            self.logger.info(f"Loading ds_resampled from: {zarr_path}")
            if storage == 'auto':
                storage = 'gcs' if zarr_path.startswith('gs://') else 'local'
            ds_resampled = load_dataset_zarr(zarr_path, storage=storage)
        
        # Step 2: Calculate MPC date range
        self.logger.info("Calculating MPC date range from ds_resampled...")
        start_date, end_date = calculate_mpc_date_range(
            ds_resampled,
            years_back=years_back,
            days_offset=days_offset
        )
        mpc_date_range = [start_date, end_date]
        self.logger.info(f"MPC date range: {mpc_date_range}")
        
        # Step 3: Load AOI and determine approach
        self.logger.info("Loading AOI and determining approach...")
        aoi_path = self.config.get('AOI_path')
        if not aoi_path:
            raise ValueError("AOI_path not found in config")
        
        # Step 4: Load AOI using DataUtils (returns original CRS, typically WGS84)
        d = DataUtils(self.config, use_gee=True)
        aoi_gpd, aoi_ee = d.load_geodataframe_gee(aoi_path)
        
        # Convert to output_crs only for area calculation (not for visualization)
        output_crs = self.config.get('output_crs', 'EPSG:4326')
        if ':' in output_crs:
            epsg_code = int(output_crs.split(':')[-1])
        else:
            epsg_code = int(output_crs)
        
        # Calculate area in output_crs (UTM) for accurate area calculation
        aoi_gpd_utm = aoi_gpd.to_crs(epsg=epsg_code)
        aoi_ha = aoi_gpd_utm.geometry.area.sum() / 10000
        
        # Determine approach based on AOI size
        use_gee = aoi_ha > aoi_size_threshold_ha
        if use_gee:
            self.logger.info(f"AOI area ({aoi_ha:.2f} ha) is too big, using GEE approach when available and mix with STAC MPC")
        else:
            self.logger.info(f"AOI area ({aoi_ha:.2f} ha) is small, using STAC (local process xarray) approach")
        
        # Step 5: Prepare config for STAC search
        config_for_stac = self.config.copy()
        config_for_stac['date_range_mpc'] = mpc_date_range
        config_for_stac['collection_mpc'] = config_for_stac.get('collection_mpc', 'sentinel-2-l2a')
        if 'url_satellite_cloud' not in config_for_stac:
            config_for_stac['url_satellite_cloud'] = 'https://planetarycomputer.microsoft.com/api/stac/v1'
        
        # Step 6: Get bounding box (use original CRS from load_geodataframe_gee, typically WGS84)
        # STAC data is in WGS84, so AOI should also be in WGS84 for visualization
        bbox = box(*aoi_gpd.total_bounds)
        
        # Step 7: Search for duplicate STAC scenes
        self.logger.info("Searching for duplicate STAC scenes...")
        selected_scene_data = search_and_analyze_duplicate_stac_scenes(
            config=config_for_stac,
            bbox=bbox,
            max_examples=max_examples,
            max_retries=max_retries
        )
        
        # Step 8: Create visualization if requested
        # Use original AOI (WGS84) - no conversion needed, matches STAC CRS
        visualization = None
        if create_visualization:
            self.logger.info("Creating visualization map...")
            print("Creating visualization map...")
            try:
                visualization = create_unified_stac_visualization(
                    gdf=aoi_gpd,  # Use original CRS (WGS84), matches STAC data
                    bbox=bbox,
                    scene_data=selected_scene_data,
                    save_map=save_map,
                    map_filename=map_filename
                )
                self.logger.info("✅ Visualization created")
                print("✅ Visualization created")
                self.logger.info(f"Visualization type: {type(visualization)}")
                print(f"Visualization type: {type(visualization)}")
                if visualization is None:
                    self.logger.warning("⚠️ Visualization is None after creation!")
                    print("⚠️ Visualization is None after creation!")
            except Exception as e:
                self.logger.error(f"Error creating visualization: {e}", exc_info=True)
                print(f"❌ Error creating visualization: {e}")
                import traceback
                traceback.print_exc()
        
        # Step 9: Generate Planetary Computer URLs
        mpc_urls = self.get_planetary_computer_urls(bbox)
        
        # Step 10: Prepare results
        results = {
            'use_gee': use_gee,
            'aoi_ha': aoi_ha,
            'mpc_date_range': mpc_date_range,
            'selected_scene_data': selected_scene_data,
            'visualization': visualization,  # folium.Map object - display directly or save to HTML
            'bbox': bbox,
            'aoi_gpd': aoi_gpd,
            'aoi_ee': aoi_ee,
            'ds_resampled': ds_resampled,
            'mpc_urls': mpc_urls
        }
        
        self.logger.info("=" * 60)
        self.logger.info("✅ MPC Eligibility Analysis Complete")
        self.logger.info("=" * 60)
        
        return results
    
    def get_planetary_computer_urls(self, bbox) -> Dict[str, str]:
        """
        Generate Microsoft Planetary Computer Explorer URLs for the given bounding box.
        
        These URLs use the correct format that Microsoft Planetary Computer Explorer
        actually uses, allowing direct viewing of satellite data.
        
        Args:
            bbox: shapely.geometry.box - Bounding box for the area of interest
            
        Returns:
            Dictionary with URL names as keys and URLs as values:
            - "🛰️ Sentinel-2 L2A (Natural Color)"
            - "🛰️ Sentinel-2 L2A (False Color)"
            - "🛰️ Landsat 8-9 (Natural Color)"
        """
        # Get center coordinates
        center_lat = (bbox.bounds[1] + bbox.bounds[3]) / 2
        center_lon = (bbox.bounds[0] + bbox.bounds[2]) / 2
        
        # URLs in the correct format (based on Microsoft Planetary Computer Explorer format)
        urls = {
            "🛰️ Sentinel-2 L2A (Natural Color)": f"https://planetarycomputer.microsoft.com/explore?c={center_lon}%2C{center_lat}&z=12&v=2&d=sentinel-2-l2a&s=false%3A%3A100%3A%3Atrue&ae=0&sr=desc&m=Most+recent+%28low+cloud%29&r=Natural+color",
            "🛰️ Sentinel-2 L2A (False Color)": f"https://planetarycomputer.microsoft.com/explore?c={center_lon}%2C{center_lat}&z=12&v=2&d=sentinel-2-l2a&s=false%3A%3A100%3A%3Atrue&ae=0&sr=desc&m=Most+recent+%28low+cloud%29&r=Short+wave+infrared",
            "🛰️ Landsat 8-9 (Natural Color)": f"https://planetarycomputer.microsoft.com/explore?c={center_lon}%2C{center_lat}&z=12&v=2&d=landsat-c2-l2&s=false%3A%3A100%3A%3Atrue&ae=0&sr=desc&m=Most+recent+%28low+cloud%29&r=Natural+color",
        }
        
        # Print formatted output
        self.logger.info("🔗 MICROSOFT PLANETARY COMPUTER URLs")
        self.logger.info("=" * 60)
        self.logger.info(f"📍 Your area center: {center_lat:.4f}, {center_lon:.4f}")
        self.logger.info(f"📍 Bbox: {bbox.bounds}")
        
        print("🔗 MICROSOFT PLANETARY COMPUTER URLs")
        print("=" * 60)
        print(f"📍 Your area center: {center_lat:.4f}, {center_lon:.4f}")
        print(f"📍 Bbox: {bbox.bounds}")
        print()
        
        for name, url in urls.items():
            self.logger.info(f"{name}: {url}")
            print(f"{name}:")
            print(f"  {url}")
            print()
        
        print("💡 These URLs use the correct format and should show satellite data directly!")
        print("🎯 Based on your working URL format from manual interaction")
        
        return urls
    
    def get_satellite_mpc(
        self,
        bbox: Optional[Any] = None,
        date_range_mpc: Optional[List[str]] = None,
        utm_crs: Optional[str] = None,
        pixel_scale: Optional[int] = None,
        zarr_path: Optional[str] = None,
        chunk_sizes: Optional[Dict[str, int]] = None,
        compression: str = 'lz4',
        compression_level: int = 1,
        overwrite_zarr: bool = False,
        save_to_zarr: bool = True,
        storage: str = 'auto',
        show_progress: bool = True,
        gee_compatible: bool = True
    ) -> 'xr.Dataset':
        """
        Download and process satellite data from Microsoft Planetary Computer (MPC) using STAC.
        
        This method:
        1. Searches for satellite data using STAC with the provided date range
        2. Downloads and processes the data (cloud masking, band renaming)
        3. Reprojects to UTM CRS (if specified or auto-calculated)
        4. Saves to zarr (local or GCS) if requested
        
        Supports multiple satellite types (Sentinel-2, Landsat, etc.) based on configuration.
        
        Args:
            bbox: Bounding box (shapely.geometry.box). If None, uses bbox from get_ds_sampled_mpc results.
            date_range_mpc: Date range for STAC search [start_date, end_date] (e.g., ['2017-01-01', '2024-12-31']).
                If None, uses date_range_mpc from get_ds_sampled_mpc results.
            utm_crs: Target UTM CRS (e.g., 'EPSG:32749'). If None, auto-calculates from AOI.
            pixel_scale: Pixel scale in meters for reprojection. If None, uses default from config (10m for Sentinel-2, 30m for Landsat).
            zarr_path: Output zarr path. If None, auto-generates from config.
            chunk_sizes: Chunk sizes for zarr storage. Default: {'time': 10, 'x': 512, 'y': 512}.
            compression: Compression algorithm ('lz4', 'blosc', 'zstd', 'gzip', 'zlib', or None). Default 'lz4'.
            compression_level: Compression level (1-9). Default 1.
            overwrite_zarr: Whether to overwrite existing zarr store. Default False.
            save_to_zarr: Whether to save to zarr. Default True.
            storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
            show_progress: Whether to show progress bars. Default True.
            gee_compatible: Whether to save in GEE-compatible format (Zarr v2, consolidated metadata).
                Default True. When True, ensures compatibility with ee.ImageCollection.loadZarrV2Array.
        
        Returns:
            xr.Dataset: Processed satellite data in UTM projection
        
        Raises:
            ValueError: If bbox or date_range_mpc not provided and not available from previous results.
        """
        try:
            import xarray as xr
            import geopandas as gpd
            import rasterio
            from shapely.geometry import box
            from ..stac.stac_processor import STACProcessor
            from ..utils.zarr_utils import save_dataset_efficient_zarr, load_dataset_zarr
            from ..utils.gee_processing import calculate_utm_crs, get_pixel_scale
        except ImportError as e:
            raise DependencyError(f"Required dependencies not available: {e}")
        
        self.logger.info("=" * 60)
        self.logger.info("🛰️  DOWNLOADING SATELLITE DATA FROM MPC")
        self.logger.info("=" * 60)
        print("🛰️  DOWNLOADING SATELLITE DATA FROM MPC")
        print("=" * 60)
        
        # Step 1: Get bbox and date_range_mpc
        if bbox is None:
            # Try to get from previous results if available
            if hasattr(self, '_last_mpc_results') and 'bbox' in self._last_mpc_results:
                bbox = self._last_mpc_results['bbox']
                self.logger.info("Using bbox from previous get_ds_sampled_mpc results")
                print("📦 Using bbox from previous get_ds_sampled_mpc results")
            else:
                raise ValueError("bbox is required. Provide bbox parameter or run get_ds_sampled_mpc first.")
        
        if date_range_mpc is None:
            # Try to get from previous results if available
            if hasattr(self, '_last_mpc_results') and 'mpc_date_range' in self._last_mpc_results:
                date_range_mpc = self._last_mpc_results['mpc_date_range']
                self.logger.info("Using date_range_mpc from previous get_ds_sampled_mpc results")
                print("📅 Using date_range_mpc from previous get_ds_sampled_mpc results")
            else:
                raise ValueError("date_range_mpc is required. Provide date_range_mpc parameter or run get_ds_sampled_mpc first.")
        
        # Format date range for STAC (e.g., "2017-01-01/2024-12-31")
        if isinstance(date_range_mpc, list) and len(date_range_mpc) == 2:
            datetime_range_str = f"{date_range_mpc[0]}/{date_range_mpc[1]}"
        else:
            datetime_range_str = str(date_range_mpc)
        
        self.logger.info(f"Date range: {datetime_range_str}")
        print(f"📅 Date range: {datetime_range_str}")
        
        # Step 2: Check if zarr already exists
        if zarr_path and save_to_zarr and not overwrite_zarr:
            try:
                ds = load_dataset_zarr(zarr_path, storage=storage)
                self.logger.info(f"✅ Loaded existing dataset from zarr: {zarr_path}")
                print(f"✅ Loaded existing dataset from zarr: {zarr_path}")
                return ds
            except (FileNotFoundError, Exception) as e:
                self.logger.info(f"Zarr not found or error loading: {e}. Proceeding with download...")
                print(f"📥 Zarr not found. Proceeding with download...")
        
        # Step 3: Initialize STAC processor with datetime override from date_range_mpc
        # Pass date_range_mpc as datetime_override to override config date_start_end
        stac_processor = STACProcessor(
            self.config_manager,  # Pass ConfigManager instance, not dict
            datetime_override=date_range_mpc,  # Use dynamic date_range_mpc instead of config
            resolution_override=None  # Use resolution_satellite from config
        )
        
        # Step 4: Process satellite data (download, cloud masking, band renaming)
        self.logger.info("Processing satellite data from STAC...")
        print("📡 Processing satellite data from STAC...")
        ds_stac = stac_processor.process_satellite_data(
            bbox=bbox,
            out_path=None,  # We'll handle zarr saving ourselves
            show_progress=show_progress,
            datetime_range=datetime_range_str
        )
        
        self.logger.info(f"✅ STAC data processed: {dict(ds_stac.sizes)}")
        print(f"✅ STAC data processed: {dict(ds_stac.sizes)}")
        
        # Step 5: Reproject to UTM if needed
        if utm_crs is None:
            # First try to use output_crs from config
            output_crs = self.config.get('output_crs')
            if output_crs:
                utm_crs = output_crs
                utm_epsg = int(utm_crs.split(':')[-1]) if ':' in utm_crs else int(utm_crs)
                self.logger.info(f"Using output_crs from config: {utm_crs}")
                print(f"🗺️  Using output_crs from config: {utm_crs}")
            else:
                # Calculate UTM CRS from bbox as fallback
                aoi_gdf = gpd.GeoDataFrame(geometry=[bbox], crs='EPSG:4326')
                utm_crs, utm_epsg, hemisphere = calculate_utm_crs(aoi_gdf)
                self.logger.info(f"Auto-calculated UTM CRS: {utm_crs}")
                print(f"🗺️  Auto-calculated UTM CRS: {utm_crs}")
        else:
            utm_epsg = int(utm_crs.split(':')[-1]) if ':' in utm_crs else int(utm_crs)
        
        if pixel_scale is None:
            pixel_scale = get_pixel_scale(self.config.get('I_satellite', 'Sentinel'))
            self.logger.info(f"Using pixel scale: {pixel_scale}m")
            print(f"📏 Pixel scale: {pixel_scale}m")
        
        # Check current CRS
        current_crs = ds_stac.rio.crs if hasattr(ds_stac, 'rio') and hasattr(ds_stac.rio, 'crs') else None
        if current_crs is None:
            # Try to get from spatial_ref coordinate
            if 'spatial_ref' in ds_stac.coords:
                current_crs = ds_stac.spatial_ref.attrs.get('crs_wkt', None)
        
        # Reproject if needed
        if current_crs is None or str(current_crs) != utm_crs:
            self.logger.info(f"Reprojecting from {current_crs} to {utm_crs}...")
            print(f"🔄 Reprojecting to {utm_crs}...")
            
            try:
                # Use rioxarray for reprojection
                if not hasattr(ds_stac, 'rio'):
                    import rioxarray
                    # Set CRS if not already set
                    if current_crs is None:
                        # Try to infer from coordinates
                        if 'spatial_ref' in ds_stac.coords:
                            ds_stac = ds_stac.rio.write_crs(ds_stac.spatial_ref.attrs.get('crs_wkt', 'EPSG:4326'))
                        else:
                            # Default to WGS84 if unknown
                            ds_stac = ds_stac.rio.write_crs('EPSG:4326')
                
                # Reproject to UTM
                ds_utm = ds_stac.rio.reproject(
                    dst_crs=utm_crs,
                    resolution=pixel_scale,
                    resampling=rasterio.enums.Resampling.bilinear
                )
                
                self.logger.info(f"✅ Reprojected to {utm_crs}")
                print(f"✅ Reprojected to {utm_crs}")
                ds_stac = ds_utm
            except Exception as e:
                self.logger.warning(f"Reprojection failed: {e}. Using original CRS.")
                print(f"⚠️  Reprojection failed: {e}. Using original CRS.")
        
        # Step 6: Save to zarr if requested
        if save_to_zarr:
            if zarr_path is None:
                # Auto-generate zarr path
                project_name = self.config.get('project_name', 'forestry_project')
                if storage == 'gcs' or (storage == 'auto' and os.getenv('GCS_ZARR_DIR')):
                    gcs_dir = os.getenv('GCS_ZARR_DIR', 'gs://remote_sensing_saas')
                    zarr_path = f"{gcs_dir}/{project_name}/satellite_mpc.zarr"
                else:
                    zarr_path = f"data/satellite_mpc/{project_name}_satellite_mpc.zarr"
            
            self.logger.info(f"Saving to zarr: {zarr_path}")
            print(f"💾 Saving to zarr: {zarr_path}")
            
            if chunk_sizes is None:
                chunk_sizes = {'time': 10, 'x': 512, 'y': 512}
            
            save_dataset_efficient_zarr(
                ds=ds_stac,
                zarr_path=zarr_path,
                chunk_sizes=chunk_sizes,
                compression=compression,
                compression_level=compression_level,
                overwrite=overwrite_zarr,
                storage=storage,
                gee_compatible=gee_compatible  # Default True - ensures GEE compatibility
            )
            
            self.logger.info("✅ Dataset saved to zarr")
            print("✅ Dataset saved to zarr")
        
        self.logger.info("=" * 60)
        self.logger.info("✅ MPC SATELLITE DATA DOWNLOAD COMPLETE")
        self.logger.info("=" * 60)
        print("=" * 60)
        print("✅ MPC SATELLITE DATA DOWNLOAD COMPLETE")
        print("=" * 60)
        
        return ds_stac
    
    def check_zarr_gee_compatibility(
        self,
        zarr_path: str,
        data_var: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if a zarr store is compatible with Google Earth Engine.
        
        GEE requires zarr URIs to end with '/.zarray' for individual arrays.
        This method checks the zarr structure and provides compatibility information.
        
        Args:
            zarr_path: Path to zarr store (local or GCS URI like gs://bucket/path.zarr)
            data_var: Optional data variable name to check specific array
        
        Returns:
            Dictionary with compatibility information:
                - 'is_gee_compatible': bool - Whether the path format is GEE-compatible
                - 'zarr_path': str - Original zarr path
                - 'gee_uri': str - GEE-compatible URI (if applicable)
                - 'is_gcs': bool - Whether path is a GCS URI
                - 'has_zarray': bool - Whether .zarray file exists
                - 'recommendations': List[str] - Recommendations for GEE compatibility
                - 'available_variables': List[str] - Available data variables (if zarr can be loaded)
        
        Example:
            >>> forestry = ForestryCarbonARR(config_path='config.json')
            >>> result = forestry.check_zarr_gee_compatibility('gs://bucket/data.zarr', 'NDVI')
            >>> print(result['gee_uri'])
            'gs://bucket/data.zarr/NDVI/.zarray'
        """
        from ..utils.zarr_utils import (
            check_zarr_gee_compatibility,
            get_gee_zarr_uri,
            list_zarr_variables
        )
        
        # Check compatibility
        result = check_zarr_gee_compatibility(zarr_path, data_var=data_var)
        
        # Try to list available variables
        try:
            variables = list_zarr_variables(zarr_path, storage='auto')
            result['available_variables'] = variables
            if variables and not data_var:
                result['recommendations'].append(
                    f"Available variables: {', '.join(variables)}. "
                    f"Specify a variable name to get the correct GEE URI."
                )
        except Exception as e:
            result['available_variables'] = []
            result['recommendations'].append(f"Could not load zarr to list variables: {e}")
        
        return result
    
    def get_gee_zarr_uri(
        self,
        zarr_path: str,
        data_var: Optional[str] = None
    ) -> str:
        """
        Convert zarr store path to GEE-compatible URI.
        
        GEE requires zarr URIs to end with '/.zarray' for individual arrays.
        
        Args:
            zarr_path: Path to zarr store (local or GCS URI like gs://bucket/path.zarr)
            data_var: Optional data variable name. If provided, returns URI for that specific array.
                      If None and dataset has multiple variables, you need to specify the variable.
        
        Returns:
            GEE-compatible URI ending with '/.zarray'
        
        Example:
            >>> forestry = ForestryCarbonARR(config_path='config.json')
            >>> uri = forestry.get_gee_zarr_uri('gs://bucket/data.zarr', 'NDVI')
            >>> print(uri)
            'gs://bucket/data.zarr/NDVI/.zarray'
        """
        from ..utils.zarr_utils import get_gee_zarr_uri
        return get_gee_zarr_uri(zarr_path, data_var=data_var)
    
    def list_zarr_variables(
        self,
        zarr_path: str,
        storage: str = 'auto'
    ) -> list:
        """
        List all data variables in a zarr store.
        
        Useful for determining which variables are available for GEE export.
        
        Args:
            zarr_path: Path to zarr store (local or GCS URI)
            storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
        
        Returns:
            List of data variable names
        
        Example:
            >>> forestry = ForestryCarbonARR(config_path='config.json')
            >>> vars = forestry.list_zarr_variables('gs://bucket/data.zarr')
            >>> print(vars)
            ['NDVI', 'EVI', 'blue', 'green']
        """
        from ..utils.zarr_utils import list_zarr_variables
        return list_zarr_variables(zarr_path, storage=storage)
    
    def convert_zarr_for_gee(
        self,
        zarr_path: str,
        output_path: Optional[str] = None,
        data_var: Optional[str] = None,
        storage: str = 'auto',
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Convert zarr dataset to GEE-compatible format.
        
        This function ensures the zarr meets all GEE requirements:
        - Adds _ARRAY_DIMENSIONS metadata
        - Uses GEE-compatible compression (lz4)
        - Creates consolidated metadata (.zmetadata)
        - Ensures CRS information is present
        - Validates spatial dimensions (Y, X as last two)
        
        Args:
            zarr_path: Path to input zarr store
            output_path: Path to output GEE-compatible zarr. If None, overwrites input.
            data_var: Optional data variable name to export. If None, exports all variables.
            storage: Storage type ('auto', 'local', 'gcs'). Default 'auto'.
            overwrite: Whether to overwrite existing output. Default False.
        
        Returns:
            Dictionary with conversion results:
                - 'success': bool - Whether conversion was successful
                - 'output_path': str - Path to converted zarr
                - 'gee_uri': str - GEE-compatible URI
                - 'warnings': List[str] - Any warnings during conversion
                - 'requirements_met': Dict[str, bool] - Which GEE requirements are met
        
        Example:
            >>> forestry = ForestryCarbonARR(config_path='config.json')
            >>> result = forestry.convert_zarr_for_gee(
            ...     'gs://bucket/data.zarr',
            ...     output_path='gs://bucket/data_gee.zarr',
            ...     data_var='NDVI'
            ... )
            >>> print(result['gee_uri'])
            'gs://bucket/data_gee.zarr/NDVI/.zarray'
        """
        from ..utils.zarr_utils import convert_zarr_for_gee
        return convert_zarr_for_gee(
            zarr_path=zarr_path,
            output_path=output_path,
            data_var=data_var,
            storage=storage,
            overwrite=overwrite
        )
    
    def __repr__(self) -> str:
        return f"ForestryCarbonARR(gee_forestry_available={self._gee_forestry_available}, strategy={self._import_strategy})"

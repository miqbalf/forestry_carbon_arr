"""
Main Forestry Carbon ARR class for GEE_notebook_Forestry integration management.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

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
            self.logger.info("Initializing Google Earth Engine...")
            from dotenv import load_dotenv
            load_dotenv()
            
            # Try to import gee_lib auth if available
            try:
                # Try gee_lib auth (container environment)
                try:
                    from gee_lib.osi.auth import initialize_gee
                    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
                    if project_id:
                        initialize_gee(project_id=project_id, use_service_account=True)
                        self.logger.info("GEE initialized successfully using gee_lib auth")
                    else:
                        raise ValueError("GOOGLE_CLOUD_PROJECT not set")
                except ImportError:
                    # Fallback to direct GEE initialization
                    try:
                        ee.Initialize()
                        self.logger.info("GEE initialized directly")
                    except Exception as init_error:
                        self.logger.error(f"Failed to initialize GEE: {init_error}")
                        raise ForestryCarbonError(f"Failed to initialize GEE: {init_error}")
            except Exception as e:
                self.logger.warning(f"Could not use gee_lib auth: {e}. Attempting direct GEE initialization...")
                # Fallback to direct initialization
                try:
                    ee.Initialize()
                    self.logger.info("GEE initialized directly (fallback)")
                except Exception as init_error:
                    self.logger.error(f"Failed to initialize GEE: {init_error}")
                    raise ForestryCarbonError(f"Failed to initialize GEE: {init_error}")
            
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
    
    def __repr__(self) -> str:
        return f"ForestryCarbonARR(gee_forestry_available={self._gee_forestry_available}, strategy={self._import_strategy})"

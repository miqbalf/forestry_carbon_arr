"""
Example usage of Forestry ARR Eligibility workflow.

This script demonstrates how to use the run_eligibility method
to run the complete forestry carbon analysis workflow.
"""

import json
import os
from forestry_carbon_arr import ForestryCarbonARR

def main():
    """
    Example usage of Forestry ARR Eligibility workflow.
    """
    # Initialize Forestry Carbon ARR
    forestry = ForestryCarbonARR()
    
    # Check if GEE_notebook_Forestry is available
    if not forestry.gee_forestry_available:
        print("ERROR: GEE_notebook_Forestry is not available!")
        print("Please ensure GEE_notebook_Forestry is properly set up.")
        return
    
    # Load configuration (example - adjust paths as needed)
    config_path = os.path.join(os.getcwd(), 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Example configuration - adjust paths and parameters as needed
        config = {
            'AOI_path': './00_input/LPHD Belaban Rayak.shp',
            'OID_field_name': 'id',
            'input_training': './00_input/traning_point_merged_belaban.shp',
            'I_satellite': 'Sentinel',  # 'Sentinel', 'Landsat', or 'Planet'
            'date_start_end': ['2024-1-1', '2024-5-31'],
            'project_name': 'belaban_rayak',
            'cloud_cover_threshold': 60,
            'region': 'asia',
            'crs_input': 'EPSG:4326',
            'algo_ml_selected': 'gbm',  # 'rf', 'svm', 'gbm', or 'cart'
            'fcd_selected': 21,  # 11, 21, 12, or 22
            'high_forest': 65,
            'yrf_forest': 45,
            'shrub_grass': 45,
            'open_land': 30,
            'year_start_loss': 14,
            'tree_cover_forest': 30,
            'pixel_number': 3,
            'pca_scaling': 1,
            'super_pixel_size': 3,
            'band_name_image': 'Class',
            'ndwi_hi_sentinel': 0.05,
            'ndwi_hi_landsat': 0.1,
            'ndwi_hi_planet': -0.2,
            'IsThermal': False  # Set to True for thermal band processing
        }
        print(f"Using example configuration. Adjust paths in config before running!")
    
    # Run the workflow
    try:
        print("Starting Forestry ARR Eligibility analysis...")
        result = forestry.run_eligibility(
            config=config,
            use_gee=True  # Set to False for STAC (not yet implemented)
        )
        
        # Access results
        final_zone = result['final_zone']
        intermediate_results = result['intermediate_results']
        visualization_params = result['visualization_params']
        updated_config = result['config']
        
        print("\n✅ Workflow completed successfully!")
        print(f"Final zone type: {type(final_zone)}")
        print(f"Algorithm used: {intermediate_results['algo_ml_selected']}")
        
        # Access visualization parameters
        print("\n📊 Visualization Parameters Available:")
        metadata = visualization_params.get('_metadata', {})
        print(f"  Mosaic: {metadata.get('mosaic', 'N/A')}")
        print(f"  FCD1_1: {metadata.get('FCD1_1', 'N/A')}")
        print(f"  FCD2_1: {metadata.get('FCD2_1', 'N/A')}")
        print(f"  Land Cover: {metadata.get('land_cover', 'N/A')}")
        print(f"  Zone: {metadata.get('zone', 'N/A')}")
        
        # Access training points information
        print("\n📍 Training Points Information:")
        training_info = intermediate_results['training_points_info']
        print(f"  Before validation: {training_info['num_points_before_validation']} points")
        print(f"  After validation: {training_info['num_points_after_validation']} points")
        print(f"  Unique classes: {training_info['unique_classes']}")
        
        # Access actual ML training and validation points (after split/stratification)
        ml_training = intermediate_results['ml_training_points']
        ml_validation = intermediate_results['ml_validation_points']
        print(f"\n  ML Training points: {ml_training.size().getInfo()} points")
        print(f"  ML Validation points: {ml_validation.size().getInfo()} points")
        
        # Example: Use visualization parameters (now direct access, no nesting)
        # vis_param_mosaic = visualization_params['mosaic']
        # vis_param_fcd1_1 = visualization_params['FCD1_1']
        # vis_param_fcd2_1 = visualization_params['FCD2_1']
        # vis_param_land_cover = visualization_params['land_cover']
        # vis_param_zone = visualization_params['zone']
        
        # Example: Use training/validation points for confusion matrix
        # from gee_lib.osi.ml.main import LandcoverML
        # lc = LandcoverML(...)
        # confusion_matrix = lc.matrix_confusion(
        #     intermediate_results['selected_image_lc'],
        #     ml_validation,
        #     intermediate_results['algo_ml_selected']
        # )
        
        # The final_zone is an Earth Engine Image object
        # You can now:
        # 1. Export it using ee.batch.Export
        # 2. Visualize it using geemap with vis_param_zone
        # 3. Use it for further analysis
        # 4. Generate confusion matrix using ml_validation_points
        
        return result
        
    except Exception as e:
        print(f"ERROR: Workflow failed: {e}")
        raise

if __name__ == '__main__':
    main()


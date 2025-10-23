"""
Default configuration for Forestry Carbon ARR library.
"""

DEFAULT_CONFIG = {
    # Project configuration
    "project": {
        "name": "forestry_carbon_project",
        "region": "global",
        "description": "Forestry Carbon ARR Analysis Project",
        "version": "1.0.0"
    },
    
    # Google Earth Engine configuration
    "gee": {
        "project_id": None,  # Will be set by user
        "service_account": None,  # Optional
        "initialize": True,
        "max_pixels": 1e13,
        "scale": 30,
        "crs": "EPSG:4326"
    },
    
    # Satellite data configuration
    "satellite": {
        "provider": "Sentinel",  # Sentinel, Planet, Landsat
        "date_range": ["2024-01-01", "2024-12-31"],
        "cloud_cover_threshold": 40,
        "bands": {
            "Sentinel": ["B2", "B3", "B4", "B8", "B11", "B12"],
            "Planet": ["red", "green", "blue", "nir"],
            "Landsat": ["B2", "B3", "B4", "B5", "B6", "B7"]
        },
        "composite_method": "median",
        "mask_clouds": True
    },
    
    # Machine Learning configuration
    "ml": {
        "algorithm": "gbm",  # gbm, random_forest, svm
        "training_samples": 1000,
        "validation_split": 0.2,
        "cross_validation": True,
        "n_folds": 5,
        "random_state": 42,
        "hyperparameter_tuning": True
    },
    
    # Forest Canopy Density configuration
    "fcd": {
        "method": "pca",  # pca, ndvi, custom
        "thresholds": {
            "high_forest": 65,
            "yrf_forest": 55,
            "shrub_grass": 35,
            "open_land": 30
        },
        "apply_smoothing": True,
        "smoothing_kernel": 3
    },
    
    # Land cover classification
    "classification": {
        "classes": {
            "1": "forest_trees",
            "2": "shrubland", 
            "3": "grassland",
            "4": "openland",
            "5": "waterbody_wet_area",
            "6": "plantation",
            "7": "infrastructure",
            "8": "oil_palm",
            "9": "cropland",
            "10": "waterbody",
            "11": "wetlands",
            "12": "forest_trees_regrowth",
            "13": "historical_treeloss_10years",
            "14": "paddy_irrigated"
        },
        "palette": {
            "1": "#83ff5a",   # forest_trees
            "2": "#ffe3b3",   # shrubland
            "3": "#ffff33",   # grassland
            "4": "#f89696",   # openland
            "5": "#1900ff",   # waterbody_wet_area
            "6": "#e6e6fa",   # plantation
            "7": "#FFFFFF",   # infrastructure
            "8": "#4B0082",   # oil_palm
            "9": "#8B4513",   # cropland
            "10": "#87CEEB",  # waterbody
            "11": "#2F4F4F",  # wetlands
            "12": "#ADFF2F",  # forest_trees_regrowth
            "13": "#8B0000",  # historical_treeloss_10years
            "14": "#DAA520"   # paddy_irrigated
        }
    },
    
    # Output configuration
    "output": {
        "format": "geotiff",  # geotiff, netcdf, shapefile
        "compression": "lzw",
        "nodata_value": -9999,
        "save_intermediate": True,
        "output_directory": "./outputs",
        "create_visualizations": True,
        "export_to_gee": False
    },
    
    # Processing configuration
    "processing": {
        "max_workers": 4,
        "chunk_size": 1000,
        "memory_limit": "8GB",
        "use_gpu": False,
        "cache_results": True,
        "cache_directory": "./cache"
    },
    
    # Dependencies configuration
    "dependencies": {
        "gee_forestry_path": None,  # Will be auto-detected
        "auto_setup": True,
        "check_dependencies": True,
        "install_missing": False
    },
    
    # Logging configuration
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "file": None,  # Log to file if specified
        "console": True
    },
    
    # Validation configuration
    "validation": {
        "validate_aoi": True,
        "validate_training_data": True,
        "validate_outputs": True,
        "strict_mode": False
    }
}

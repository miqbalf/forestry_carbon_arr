"""
Configuration loading utilities for Forestry Carbon ARR library.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Utility class for loading configuration from various sources.
    """
    
    @staticmethod
    def load_from_file(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    @staticmethod
    def load_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration dictionary (copy)
        """
        return config_dict.copy()
    
    @staticmethod
    def load_from_env() -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Returns:
            Configuration dictionary from environment
        """
        import os
        
        config = {}
        
        # GEE configuration
        if os.environ.get('GEE_PROJECT_ID'):
            config['gee'] = {'project_id': os.environ.get('GEE_PROJECT_ID')}
        
        # GEE Forestry path
        if os.environ.get('GEE_FORESTRY_PATH'):
            config['dependencies'] = {'gee_forestry_path': os.environ.get('GEE_FORESTRY_PATH')}
        
        return config
    
    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        def deep_merge(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        result = {}
        for config in configs:
            result = deep_merge(result, config)
        
        return result

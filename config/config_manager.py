"""
Configuration manager for Forestry Carbon ARR library.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Union, Optional
import logging

from .default_config import DEFAULT_CONFIG
from ..utils.path_resolver import PathResolver

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages configuration for Forestry Carbon ARR analysis.
    
    This class handles:
    - Loading configuration from files or dictionaries
    - Merging with default configuration
    - Validating configuration parameters
    - Providing easy access to configuration values
    """
    
    def __init__(self, config_source: Optional[Union[str, Path, Dict]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_source: Path to config file, config dict, or None for defaults
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.path_resolver = PathResolver()
        
        # Start with default configuration
        self._config = DEFAULT_CONFIG.copy()
        
        # Load additional configuration if provided
        if config_source is not None:
            self.load_config(config_source)
    
    def load_config(self, config_source: Union[str, Path, Dict]) -> None:
        """
        Load configuration from various sources.
        
        Args:
            config_source: Path to config file or configuration dictionary
        """
        if isinstance(config_source, dict):
            self._merge_config(config_source)
        else:
            # Assume it's a file path
            config_path = self.path_resolver.resolve_config_path(config_source)
            self._load_config_file(config_path)
    
    def _load_config_file(self, config_path: Path) -> None:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    file_config = yaml.safe_load(f)
                else:
                    file_config = json.load(f)
            
            self._merge_config(file_config)
            self.logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def _normalize_flat_config(self, flat_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize flat configuration keys to nested structure.
        
        Maps legacy flat keys (from JSON files) to nested structure:
        - date_start_end → satellite.date_range
        - cloud_cover_threshold → satellite.cloud_cover_threshold
        - I_satellite → satellite.provider
        - high_forest, yrf_forest, etc. → fcd.thresholds.*
        - region → project.region
        
        Args:
            flat_config: Flat configuration dictionary
            
        Returns:
            Normalized configuration dictionary
        """
        normalized = {}
        flat_keys_to_remove = []
        
        # Mapping of flat keys to nested paths
        key_mappings = {
            'date_start_end': 'satellite.date_range',
            'cloud_cover_threshold': 'satellite.cloud_cover_threshold',
            'I_satellite': 'satellite.provider',
            'region': 'project.region',
            'high_forest': 'fcd.thresholds.high_forest',
            'yrf_forest': 'fcd.thresholds.yrf_forest',
            'shrub_grass': 'fcd.thresholds.shrub_grass',
            'open_land': 'fcd.thresholds.open_land',
        }
        
        # Apply mappings
        for flat_key, nested_path in key_mappings.items():
            if flat_key in flat_config:
                keys = nested_path.split('.')
                # Create nested structure
                current = normalized
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = flat_config[flat_key]
                flat_keys_to_remove.append(flat_key)
        
        # Copy remaining keys that don't need mapping
        for key, value in flat_config.items():
            if key not in flat_keys_to_remove:
                normalized[key] = value
        
        return normalized
    
    def _merge_config(self, new_config: Dict[str, Any]) -> None:
        """
        Merge new configuration with existing configuration.
        
        Args:
            new_config: New configuration to merge
        """
        # Normalize flat keys to nested structure
        normalized_config = self._normalize_flat_config(new_config)
        
        def deep_merge(base_dict: Dict, update_dict: Dict) -> Dict:
            """Recursively merge dictionaries."""
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        self._config = deep_merge(self._config, normalized_config)
        self.logger.debug("Configuration merged successfully")
    
    def _get_with_fallback(self, key: str, nested_path: str = None) -> Any:
        """
        Get configuration value with backward compatibility.
        
        Checks nested path first, then falls back to flat key.
        
        Args:
            key: Flat key name (e.g., 'date_start_end')
            nested_path: Nested path (e.g., 'satellite.date_range')
            
        Returns:
            Configuration value or None
        """
        if nested_path:
            value = self.get(nested_path)
            if value is not None:
                return value
        
        # Fallback to flat key
        return self._config.get(key)
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns a clean configuration without duplicate flat keys
        that have been normalized to nested structure.
        """
        config_copy = self._config.copy()
        
        # Remove flat keys that have been normalized to nested structure
        # to avoid confusion and duplicates
        flat_keys_to_remove = [
            'date_start_end',  # → satellite.date_range
            'cloud_cover_threshold',  # → satellite.cloud_cover_threshold (if in satellite section)
            'I_satellite',  # → satellite.provider
            'region',  # → project.region (if in project section)
            'high_forest',  # → fcd.thresholds.high_forest
            'yrf_forest',  # → fcd.thresholds.yrf_forest
            'shrub_grass',  # → fcd.thresholds.shrub_grass
            'open_land',  # → fcd.thresholds.open_land
        ]
        
        # Only remove if they exist in nested structure
        for key in flat_keys_to_remove:
            if key in config_copy:
                # Check if normalized version exists
                mappings = {
                    'date_start_end': 'satellite.date_range',
                    'cloud_cover_threshold': 'satellite.cloud_cover_threshold',
                    'I_satellite': 'satellite.provider',
                    'region': 'project.region',
                    'high_forest': 'fcd.thresholds.high_forest',
                    'yrf_forest': 'fcd.thresholds.yrf_forest',
                    'shrub_grass': 'fcd.thresholds.shrub_grass',
                    'open_land': 'fcd.thresholds.open_land',
                }
                nested_path = mappings.get(key)
                if nested_path and self.get(nested_path) is not None:
                    # Remove flat key if nested version exists
                    del config_copy[key]
        
        return config_copy
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Also supports backward compatibility for flat keys:
        - 'date_start_end' → checks 'satellite.date_range' first
        - 'cloud_cover_threshold' → checks 'satellite.cloud_cover_threshold' first
        - 'I_satellite' → checks 'satellite.provider' first
        - etc.
        
        Args:
            key: Configuration key (e.g., 'gee.project_id' or 'ml.algorithm' or 'date_start_end')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Backward compatibility mapping for flat keys
        flat_to_nested = {
            'date_start_end': 'satellite.date_range',
            'cloud_cover_threshold': 'satellite.cloud_cover_threshold',
            'I_satellite': 'satellite.provider',
            'region': 'project.region',
            'high_forest': 'fcd.thresholds.high_forest',
            'yrf_forest': 'fcd.thresholds.yrf_forest',
            'shrub_grass': 'fcd.thresholds.shrub_grass',
            'open_land': 'fcd.thresholds.open_land',
        }
        
        # If it's a flat key with a nested equivalent, check nested first
        if key in flat_to_nested and '.' not in key:
            nested_path = flat_to_nested[key]
            keys = nested_path.split('.')
            value = self._config
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                # Fall back to flat key
                pass
        
        # Try as nested path (dot notation)
        if '.' in key:
            keys = key.split('.')
            value = self._config
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
        
        # Try as flat key
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'gee.project_id')
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
        self.logger.debug(f"Configuration set: {key} = {value}")
    
    def save_config(self, file_path: Union[str, Path], format: str = 'json') -> None:
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2)
            
            self.logger.info(f"Configuration saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
    
    def validate_config(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Check required fields
            required_fields = [
                'project.name',
                'project.region',
                'satellite.provider'
            ]
            
            for field in required_fields:
                if self.get(field) is None:
                    self.logger.error(f"Required configuration field missing: {field}")
                    return False
            
            # Validate satellite provider
            valid_providers = ['Sentinel', 'Planet', 'Landsat']
            provider = self.get('satellite.provider')
            if provider not in valid_providers:
                self.logger.error(f"Invalid satellite provider: {provider}. Must be one of {valid_providers}")
                return False
            
            # Validate date format
            date_range = self.get('satellite.date_range')
            if date_range and len(date_range) != 2:
                self.logger.error("Date range must contain exactly 2 dates")
                return False
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_gee_config(self) -> Dict[str, Any]:
        """Get GEE-specific configuration."""
        return self.get('gee', {})
    
    def get_ml_config(self) -> Dict[str, Any]:
        """Get ML-specific configuration."""
        return self.get('ml', {})
    
    def get_satellite_config(self) -> Dict[str, Any]:
        """Get satellite-specific configuration."""
        return self.get('satellite', {})
    
    def get_project_config(self) -> Dict[str, Any]:
        """Get project-specific configuration."""
        return self.get('project', {})
    
    def update_gee_forestry_path(self, path: Union[str, Path]) -> None:
        """
        Update GEE Forestry path in configuration.
        
        Args:
            path: Path to GEE_notebook_Forestry directory
        """
        self.set('dependencies.gee_forestry_path', str(path))
        self.logger.info(f"GEE Forestry path updated: {path}")
    
    def __repr__(self) -> str:
        return f"ConfigManager(project={self.get('project.name', 'unknown')})"

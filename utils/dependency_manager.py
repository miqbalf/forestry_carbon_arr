"""
Dependency management for Forestry Carbon ARR library.
"""

import sys
import importlib
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DependencyManager:
    """
    Manages dependencies and optional imports for the Forestry Carbon ARR library.
    
    This class handles:
    - Checking for required and optional dependencies
    - Managing imports from GEE_notebook_Forestry
    - Providing graceful fallbacks when dependencies are missing
    - Validating system requirements
    """
    
    def __init__(self):
        """Initialize dependency manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Core dependencies (required)
        self.core_dependencies = {
            'numpy': 'numpy',
            'pandas': 'pandas',
            'geopandas': 'geopandas',
            'shapely': 'shapely',
            'pyproj': 'pyproj',
            'fiona': 'fiona',
            'rasterio': 'rasterio'
        }
        
        # Optional dependencies (for future use)
        self.optional_dependencies = {
            'gee': {
                'earthengine-api': 'ee',
                'geemap': 'geemap',
                'folium': 'folium'
            }
        }
        
        # GEE Forestry specific modules
        self.gee_forestry_modules = [
            'osi.image_collection.main',
            'osi.spectral_indices.spectral_analysis',
            'osi.fcd.main_fcd',
            'osi.ml.main',
            'osi.arcpy.main',
            'osi.obia.main',
            'osi.hansen.historical_loss',
            'osi.utils.main'
        ]
        
        # Cache for dependency checks
        self._dependency_cache = {}
        self._gee_forestry_cache = {}
    
    def check_core_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Check if all core dependencies are available.
        
        Returns:
            Tuple of (all_available, missing_dependencies)
        """
        missing = []
        
        for package_name, import_name in self.core_dependencies.items():
            if not self._check_import(import_name):
                missing.append(package_name)
        
        all_available = len(missing) == 0
        return all_available, missing
    
    def check_optional_dependencies(self, category: str) -> Tuple[bool, List[str]]:
        """
        Check optional dependencies for a specific category.
        
        Args:
            category: Category name (gee, ml, satellite, visualization)
            
        Returns:
            Tuple of (all_available, missing_dependencies)
        """
        if category not in self.optional_dependencies:
            raise ValueError(f"Unknown dependency category: {category}")
        
        missing = []
        deps = self.optional_dependencies[category]
        
        for package_name, import_name in deps.items():
            if not self._check_import(import_name):
                missing.append(package_name)
        
        all_available = len(missing) == 0
        return all_available, missing
    
    def check_gee_forestry_modules(self) -> Tuple[bool, List[str]]:
        """
        Check if GEE_notebook_Forestry modules are available.
        
        Returns:
            Tuple of (all_available, missing_modules)
        """
        missing = []
        
        for module_name in self.gee_forestry_modules:
            if not self._check_gee_forestry_module(module_name):
                missing.append(module_name)
        
        all_available = len(missing) == 0
        return all_available, missing
    
    def _check_import(self, import_name: str) -> bool:
        """
        Check if a module can be imported.
        
        Args:
            import_name: Name to import
            
        Returns:
            True if import successful, False otherwise
        """
        if import_name in self._dependency_cache:
            return self._dependency_cache[import_name]
        
        try:
            importlib.import_module(import_name)
            self._dependency_cache[import_name] = True
            return True
        except ImportError:
            self._dependency_cache[import_name] = False
            return False
    
    def _check_gee_forestry_module(self, module_name: str) -> bool:
        """
        Check if a GEE_notebook_Forestry module can be imported.
        
        Args:
            module_name: Module name to check
            
        Returns:
            True if import successful, False otherwise
        """
        if module_name in self._gee_forestry_cache:
            return self._gee_forestry_cache[module_name]
        
        try:
            importlib.import_module(module_name)
            self._gee_forestry_cache[module_name] = True
            return True
        except ImportError:
            self._gee_forestry_cache[module_name] = False
            return False
    
    def is_gee_available(self) -> bool:
        """Check if GEE dependencies are available."""
        available, _ = self.check_optional_dependencies('gee')
        return available
    
    
    def is_gee_forestry_available(self) -> bool:
        """Check if GEE_notebook_Forestry modules are available."""
        available, _ = self.check_gee_forestry_modules()
        return available
    
    def get_dependency_status(self) -> Dict[str, Dict[str, bool]]:
        """
        Get comprehensive dependency status.
        
        Returns:
            Dictionary with dependency status for each category
        """
        status = {}
        
        # Core dependencies
        core_available, core_missing = self.check_core_dependencies()
        status['core'] = {
            'available': core_available,
            'missing': core_missing
        }
        
        # Optional dependencies
        for category in self.optional_dependencies:
            available, missing = self.check_optional_dependencies(category)
            status[category] = {
                'available': available,
                'missing': missing
            }
        
        # GEE Forestry modules
        gee_forestry_available, gee_forestry_missing = self.check_gee_forestry_modules()
        status['gee_forestry'] = {
            'available': gee_forestry_available,
            'missing': gee_forestry_missing
        }
        
        return status
    
    def validate_dependencies(self) -> None:
        """
        Validate that all required dependencies are available.
        
        Raises:
            ImportError: If core dependencies are missing
        """
        core_available, missing = self.check_core_dependencies()
        
        if not core_available:
            missing_str = ', '.join(missing)
            raise ImportError(
                f"Missing required dependencies: {missing_str}. "
                f"Please install them using: pip install {' '.join(missing)}"
            )
        
        self.logger.info("All core dependencies are available")
    
    def get_installation_commands(self) -> Dict[str, str]:
        """
        Get installation commands for missing dependencies.
        
        Returns:
            Dictionary with installation commands for each category
        """
        commands = {}
        
        # Core dependencies
        _, core_missing = self.check_core_dependencies()
        if core_missing:
            commands['core'] = f"pip install {' '.join(core_missing)}"
        
        # Optional dependencies
        for category in self.optional_dependencies:
            _, missing = self.check_optional_dependencies(category)
            if missing:
                commands[category] = f"pip install {' '.join(missing)}"
        
        return commands
    
    def clear_cache(self) -> None:
        """Clear dependency check cache."""
        self._dependency_cache.clear()
        self._gee_forestry_cache.clear()
        self.logger.debug("Dependency cache cleared")

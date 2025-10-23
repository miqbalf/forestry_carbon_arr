"""
Import manager for GEE_notebook_Forestry integration.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from .path_resolver import PathResolver

logger = logging.getLogger(__name__)


class ImportManager:
    """
    Manages imports from GEE_notebook_Forestry with support for both local and container environments.
    
    This class handles:
    - Detecting GEE_notebook_Forestry location (side-by-side vs container)
    - Setting up proper Python path
    - Providing import strategies for different environments
    - Managing import aliases (GEE_notebook_Forestry vs gee_lib)
    """
    
    def __init__(self):
        """Initialize import manager."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.path_resolver = PathResolver()
        self._gee_forestry_path = None
        self._import_strategy = None
        self._import_aliases = {}
    
    def detect_and_setup_gee_forestry(self) -> Tuple[bool, Optional[Path], str]:
        """
        Detect GEE_notebook_Forestry and setup import strategy.
        
        Returns:
            Tuple of (success, path, strategy)
            - success: Whether GEE_notebook_Forestry was found and setup
            - path: Path to GEE_notebook_Forestry directory
            - strategy: Import strategy ('local', 'container', or 'none')
        """
        try:
            # Resolve possible paths
            possible_paths = self.path_resolver.resolve_gee_forestry_paths()
            
            for path in possible_paths:
                if path and path.exists() and (path / "osi").exists():
                    self._gee_forestry_path = path
                    
                    # Determine import strategy based on path
                    strategy = self._determine_import_strategy(path)
                    self._import_strategy = strategy
                    
                    # Setup Python path and aliases
                    self._setup_import_environment(path, strategy)
                    
                    self.logger.info(f"GEE_notebook_Forestry detected: {path} (strategy: {strategy})")
                    return True, path, strategy
            
            self.logger.warning("GEE_notebook_Forestry not found")
            return False, None, 'none'
            
        except Exception as e:
            self.logger.error(f"Failed to detect GEE_notebook_Forestry: {e}")
            return False, None, 'none'
    
    def _determine_import_strategy(self, path: Path) -> str:
        """
        Determine import strategy based on path.
        
        Args:
            path: Path to GEE_notebook_Forestry
            
        Returns:
            Import strategy ('local' or 'container')
        """
        path_str = str(path)
        
        # Container paths
        if any(container_indicator in path_str for container_indicator in [
            '/usr/src/app/gee_lib',
            '/app/gee_lib',
            '/usr/src/app/GEE_notebook_Forestry',
            '/app/GEE_notebook_Forestry'
        ]):
            return 'container'
        
        # Local development paths (side-by-side)
        if 'GEE_notebook_Forestry' in path_str:
            return 'local'
        
        # Default to local
        return 'local'
    
    def _setup_import_environment(self, path: Path, strategy: str) -> None:
        """
        Setup import environment based on strategy.
        
        Args:
            path: Path to GEE_notebook_Forestry
            strategy: Import strategy
        """
        # Add to Python path
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
            self.logger.info(f"Added {path} to Python path")
        
        # Set environment variable
        os.environ['GEE_FORESTRY_PATH'] = str(path)
        
        # Setup import aliases based on strategy
        if strategy == 'container':
            # In container, we might want to import as 'gee_lib'
            self._import_aliases = {
                'gee_lib': path.name,  # Map 'gee_lib' to actual directory name
                'GEE_notebook_Forestry': path.name
            }
        else:
            # In local development, use actual directory name
            self._import_aliases = {
                'GEE_notebook_Forestry': path.name
            }
    
    def get_import_examples(self) -> Dict[str, List[str]]:
        """
        Get import examples for different scenarios.
        
        Returns:
            Dictionary with import examples for each strategy
        """
        examples = {
            'local': [
                "# Local development (side-by-side)",
                "from GEE_notebook_Forestry.osi import *",
                "from GEE_notebook_Forestry.osi.image_collection.main import ImageCollection",
                "from GEE_notebook_Forestry.osi.fcd.main_fcd import FCDCalc",
                "from GEE_notebook_Forestry.osi.ml.main import LandcoverML"
            ],
            'container': [
                "# Container environment (gee_lib)",
                "from gee_lib.osi import *",
                "from gee_lib.osi.image_collection.main import ImageCollection", 
                "from gee_lib.osi.fcd.main_fcd import FCDCalc",
                "from gee_lib.osi.ml.main import LandcoverML"
            ],
            'none': [
                "# GEE_notebook_Forestry not available",
                "# Use basic functionality or install GEE_notebook_Forestry"
            ]
        }
        
        return examples
    
    def safe_import_osi_module(self, module_name: str) -> Tuple[Any, bool]:
        """
        Safely import an osi module.
        
        Args:
            module_name: Module name to import
            
        Returns:
            Tuple of (module, success)
        """
        try:
            # Try direct import first
            module = __import__(module_name)
            return module, True
        except ImportError as e:
            self.logger.debug(f"Failed to import {module_name}: {e}")
            return None, False
    
    def get_available_osi_modules(self) -> Dict[str, bool]:
        """
        Get list of available osi modules.
        
        Returns:
            Dictionary with module names and availability
        """
        if not self._gee_forestry_path:
            return {}
        
        # List of modules to test
        modules_to_test = [
            'osi',
            'osi.utils.main',
            'osi.image_collection.main',
            'osi.spectral_indices.spectral_analysis',
            'osi.fcd.main_fcd',
            'osi.ml.main',
            'osi.arcpy.main',
            'osi.obia.main',
            'osi.hansen.historical_loss'
        ]
        
        results = {}
        for module_name in modules_to_test:
            _, success = self.safe_import_osi_module(module_name)
            results[module_name] = success
        
        return results
    
    def get_import_info(self) -> Dict[str, Any]:
        """
        Get comprehensive import information.
        
        Returns:
            Dictionary with import information
        """
        return {
            'gee_forestry_path': str(self._gee_forestry_path) if self._gee_forestry_path else None,
            'import_strategy': self._import_strategy,
            'import_aliases': self._import_aliases,
            'python_path_added': str(self._gee_forestry_path) in sys.path if self._gee_forestry_path else False,
            'environment_variable': os.environ.get('GEE_FORESTRY_PATH'),
            'available_modules': self.get_available_osi_modules(),
            'import_examples': self.get_import_examples()
        }
    
    def create_import_guide(self) -> str:
        """
        Create an import guide for users.
        
        Returns:
            Formatted import guide
        """
        info = self.get_import_info()
        
        guide = f"""
🌳 GEE_notebook_Forestry Import Guide
{'=' * 50}

Current Setup:
- Path: {info['gee_forestry_path'] or 'Not found'}
- Strategy: {info['import_strategy'] or 'none'}
- Python Path: {'✅ Added' if info['python_path_added'] else '❌ Not added'}

Available Modules:
"""
        
        for module, available in info['available_modules'].items():
            status = "✅" if available else "❌"
            guide += f"  {status} {module}\n"
        
        guide += f"\nImport Examples:\n"
        for strategy, examples in info['import_examples'].items():
            if strategy == info['import_strategy']:
                guide += f"\n{strategy.upper()} (Current):\n"
                for example in examples:
                    guide += f"  {example}\n"
        
        return guide

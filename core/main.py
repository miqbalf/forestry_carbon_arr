"""
Main Forestry Carbon ARR class for GEE_notebook_Forestry integration management.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

try:
    from ..config import ConfigManager
    from ..exceptions import ForestryCarbonError, DependencyError
    from ..utils.dependency_manager import DependencyManager
    from ..utils.path_resolver import PathResolver
    from ..utils.import_manager import ImportManager
except ImportError:
    # For direct execution
    from config import ConfigManager
    from exceptions import ForestryCarbonError, DependencyError
    from utils.dependency_manager import DependencyManager
    from utils.path_resolver import PathResolver
    from utils.import_manager import ImportManager

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
    
    def __repr__(self) -> str:
        return f"ForestryCarbonARR(gee_forestry_available={self._gee_forestry_available}, strategy={self._import_strategy})"

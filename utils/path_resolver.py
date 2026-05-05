"""
Path resolution utilities for Forestry Carbon ARR library.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """
    Resolves paths to external dependencies and resources.
    
    This class handles:
    - Finding GEE_notebook_Forestry in various locations
    - Resolving paths in container environments
    - Handling development vs production path differences
    - Managing relative and absolute path resolution
    """
    
    def __init__(self):
        """Initialize path resolver."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def resolve_gee_forestry_paths(self) -> List[Optional[Path]]:
        """
        Resolve possible paths to GEE_notebook_Forestry directory.
        
        Priority order:
        1. Environment variable (GEE_FORESTRY_PATH)
        2. Side-by-side with forestry_carbon_arr (local development)
        3. Container paths (gee_lib)
        4. PYTHONPATH locations
        5. Common development paths
        
        Returns:
            List of possible paths (some may be None or non-existent)
        """
        paths = []
        
        # 1. Check environment variable (highest priority)
        env_path = os.environ.get('GEE_FORESTRY_PATH')
        if env_path:
            paths.append(Path(env_path))
        
        # 2. Check side-by-side with forestry_carbon_arr (local development)
        try:
            # Get the directory containing forestry_carbon_arr
            forestry_carbon_path = Path(__file__).parent.parent.parent
            side_by_side_path = forestry_carbon_path / "GEE_notebook_Forestry"
            if side_by_side_path.exists():
                paths.append(side_by_side_path)
                self.logger.info(f"Found GEE_notebook_Forestry side-by-side: {side_by_side_path}")
        except Exception:
            pass
        
        # 3. Check container paths (for Docker deployment)
        container_paths = [
            Path("/usr/src/app/gee_lib"),      # Development container
            Path("/app/gee_lib"),              # Production container
            Path("/usr/src/app/GEE_notebook_Forestry"),  # Alternative container path
            Path("/app/GEE_notebook_Forestry"),          # Alternative container path
        ]
        for container_path in container_paths:
            if container_path.exists():
                paths.append(container_path)
                self.logger.info(f"Found GEE_notebook_Forestry in container: {container_path}")
        
        # 4. Check PYTHONPATH
        for python_path in sys.path:
            if python_path and python_path != '':
                # Look for GEE_notebook_Forestry in PYTHONPATH
                gee_path = Path(python_path) / "GEE_notebook_Forestry"
                if gee_path.exists():
                    paths.append(gee_path)
                
                # Also check parent directories
                parent = Path(python_path).parent
                gee_path = parent / "GEE_notebook_Forestry"
                if gee_path.exists():
                    paths.append(gee_path)
        
        # 5. Check current working directory and parents
        current_path = Path.cwd()
        for _ in range(5):  # Check up to 5 levels up
            gee_path = current_path / "GEE_notebook_Forestry"
            if gee_path.exists():
                paths.append(gee_path)
            current_path = current_path.parent
            if current_path == current_path.parent:  # Reached root
                break
        
        # 6. Check common development paths
        home = Path.home()
        dev_paths = [
            home / "projects" / "gis-carbon-ai" / "GEE_notebook_Forestry",
            home / "gis-carbon-ai" / "GEE_notebook_Forestry",
            home / "GEE_notebook_Forestry"
        ]
        paths.extend(dev_paths)
        
        return paths
    
    def resolve_config_path(self, config_path: Union[str, Path]) -> Path:
        """
        Resolve configuration file path.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Resolved Path object
        """
        config_path = Path(config_path)
        
        # If absolute path, return as is
        if config_path.is_absolute():
            return config_path
        
        # Try relative to current working directory
        cwd_path = Path.cwd() / config_path
        if cwd_path.exists():
            return cwd_path
        
        # Try relative to forestry_carbon_arr directory
        forestry_carbon_path = Path(__file__).parent.parent
        lib_path = forestry_carbon_path / config_path
        if lib_path.exists():
            return lib_path
        
        # Try relative to user's home directory
        home_path = Path.home() / config_path
        if home_path.exists():
            return home_path
        
        # Return original path if none found (will raise error later)
        return config_path
    
    def resolve_data_path(self, data_path: Union[str, Path], 
                         base_path: Optional[Path] = None) -> Path:
        """
        Resolve data file path.
        
        Args:
            data_path: Path to data file
            base_path: Base path to search from (optional)
            
        Returns:
            Resolved Path object
        """
        data_path = Path(data_path)
        
        # If absolute path, return as is
        if data_path.is_absolute():
            return data_path
        
        # Search paths in order of preference
        search_paths = []
        
        if base_path:
            search_paths.append(base_path)
        
        search_paths.extend([
            Path.cwd(),
            Path(__file__).parent.parent,
            Path.home()
        ])
        
        for base in search_paths:
            full_path = base / data_path
            if full_path.exists():
                return full_path
        
        # Return original path if none found
        return data_path
    
    def get_container_paths(self) -> dict:
        """
        Get common container paths for different services.
        
        Returns:
            Dictionary with container path mappings
        """
        return {
            'jupyter': {
                'base': Path('/usr/src/app'),
                'notebooks': Path('/usr/src/app/notebooks'),
                'data': Path('/usr/src/app/data'),
                'gee_lib': Path('/usr/src/app/gee_lib'),
                'ex_ante': Path('/usr/src/app/ex_ante_lib')
            },
            'django': {
                'base': Path('/usr/src/app'),
                'static': Path('/var/www/static'),
                'media': Path('/var/www/media')
            },
            'fastapi': {
                'base': Path('/usr/src/app'),
                'cache': Path('/usr/src/app/cache')
            }
        }
    
    def is_container_environment(self) -> bool:
        """
        Check if running in a container environment.
        
        Returns:
            True if running in container, False otherwise
        """
        # Check for common container indicators
        container_indicators = [
            '/.dockerenv',
            '/proc/1/cgroup',
            'KUBERNETES_SERVICE_HOST' in os.environ,
            'CONTAINER' in os.environ
        ]
        
        for indicator in container_indicators:
            if isinstance(indicator, str):
                if Path(indicator).exists():
                    return True
            else:
                if indicator:
                    return True
        
        return False
    
    def get_environment_info(self) -> dict:
        """
        Get information about the current environment.
        
        Returns:
            Dictionary with environment information
        """
        return {
            'is_container': self.is_container_environment(),
            'python_path': sys.path,
            'working_directory': Path.cwd(),
            'home_directory': Path.home(),
            'environment_variables': {
                'GEE_FORESTRY_PATH': os.environ.get('GEE_FORESTRY_PATH'),
                'PYTHONPATH': os.environ.get('PYTHONPATH'),
                'CONTAINER': os.environ.get('CONTAINER')
            },
            'possible_gee_paths': [str(p) for p in self.resolve_gee_forestry_paths() if p]
        }

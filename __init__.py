"""
Forestry Carbon ARR Library

A library for managing GEE_notebook_Forestry integration in forestry carbon analysis workflows.

This library provides import and path management for integrating with GEE_notebook_Forestry,
supporting both local development (side-by-side) and container environments.
"""

__version__ = "0.1.0"
__author__ = "GIS Carbon AI Team"
__email__ = "muh.firdausiqbal@gmail.com"

# Core imports
from .core import ForestryCarbonARR
from .config import ConfigManager
from .exceptions import ForestryCarbonError, DependencyError

# Export main classes
__all__ = [
    'ForestryCarbonARR',
    'ConfigManager',
    'ForestryCarbonError',
    'DependencyError'
]

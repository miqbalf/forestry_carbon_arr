"""
Spectral Indices Formula Utilities

This module provides utilities for working with spectral indices formulas,
converting between eemont-osi format and OSI band names.
"""

import re
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Try to import ee_extra (eemont-osi)
try:
    import ee_extra.Spectral.core as spec_core
    EE_EXTRA_AVAILABLE = True
except ImportError:
    EE_EXTRA_AVAILABLE = False
    spec_core = None
    logger.warning("ee_extra not available. Install with: pip install eemont-osi")


class SpectralIndicesUtils:
    """
    Utility class for working with spectral indices formulas.
    
    Provides functions to:
    - Get spectral index formulas from eemont-osi
    - Convert formulas to OSI band names
    - Get index metadata and information
    """
    
    def __init__(self):
        """Initialize spectral indices utilities."""
        if not EE_EXTRA_AVAILABLE:
            raise ImportError(
                "ee_extra (eemont-osi) is required. Install with: pip install eemont-osi"
            )
        
        # Get all available indices dynamically
        self.indices_dict = spec_core.indices(online=False)
        self.spectral_indices_list = sorted(list(self.indices_dict.keys()))
        
        logger.info(f"Loaded {len(self.spectral_indices_list)} spectral indices from eemont-osi")
    
    def get_index_info(self, index_name: str) -> Dict[str, Any]:
        """
        Get formula and metadata for a spectral index.
        
        Parameters
        ----------
        index_name : str
            Name of the spectral index (e.g., 'NDVI', 'EVI', 'SAVI')
        
        Returns
        -------
        dict
            Dictionary containing:
            - formula: Mathematical formula string
            - long_name: Full name of the index
            - bands: List of bands used (N=NIR, R=Red, G=Green, B=Blue, S1=SWIR1, S2=SWIR2, RE1-4=Red Edge)
            - application_domain: Category (vegetation, water, burn, etc.)
            - platforms: Supported satellite platforms
            - reference: Reference URL or DOI
        """
        index_name_upper = index_name.upper()
        
        if index_name_upper not in self.indices_dict:
            available = [
                idx for idx in self.spectral_indices_list 
                if index_name.upper() in idx.upper()
            ]
            raise ValueError(
                f"Index '{index_name}' not found. "
                f"Did you mean: {available[:5] if available else 'None'}?"
            )
        
        info = self.indices_dict[index_name_upper].copy()
        return info
    
    def formula(self, index_name: str) -> str:
        """
        Get the formula for a spectral index.
        
        Parameters
        ----------
        index_name : str
            Name of the spectral index (e.g., 'NDVI', 'EVI', 'SAVI')
        
        Returns
        -------
        str
            Mathematical formula string using band abbreviations
            Band abbreviations: N (NIR), R (Red), G (Green), B (Blue), 
            S1 (SWIR1), S2 (SWIR2), RE1-4 (Red Edge 1-4)
        
        Examples
        --------
        >>> utils = SpectralIndicesUtils()
        >>> utils.formula('NDVI')
        '(N - R)/(N + R)'
        
        >>> utils.formula('EVI')
        'G * ((N - R) / (N + C1 * R - C2 * B + L))'
        """
        info = self.get_index_info(index_name)
        return info['formula']
    
    def formula_to_osi_bands(self, formula_str: str) -> str:
        """
        Convert eemont-osi formula band abbreviations to OSI band names.
        
        Mapping:
        - N (NIR) -> nir
        - R (Red) -> red
        - G (Green) -> green
        - B (Blue) -> blue
        - S1 (SWIR1) -> swir1
        - S2 (SWIR2) -> swir2
        - RE1 (Red Edge 1) -> redE1
        - RE2 (Red Edge 2) -> redE2
        - RE3 (Red Edge 3) -> redE3
        - RE4 (Red Edge 4) -> redE4
        - Variables (g, C1, C2, L, etc.) remain as-is
        
        Parameters
        ----------
        formula_str : str
            Formula string from eemont-osi (e.g., "(N - R)/(N + R)")
        
        Returns
        -------
        str
            Formula with OSI band names (e.g., "(nir - red)/(nir + red)")
        """
        # Mapping from eemont-osi abbreviations to OSI band names
        band_mapping = {
            'N': 'nir',      # Near Infrared
            'R': 'red',      # Red
            'G': 'green',    # Green
            'B': 'blue',     # Blue
            'S1': 'swir1',   # Shortwave Infrared 1
            'S2': 'swir2',   # Shortwave Infrared 2
            'RE1': 'redE1',  # Red Edge 1
            'RE2': 'redE2',  # Red Edge 2
            'RE3': 'redE3',  # Red Edge 3
            'RE4': 'redE4',  # Red Edge 4
        }
        
        # Sort by length (longest first) to avoid partial matches (e.g., RE1 before R)
        sorted_bands = sorted(band_mapping.keys(), key=len, reverse=True)
        
        result = formula_str
        
        # Replace band abbreviations with OSI names
        # Use word boundaries to avoid replacing partial matches in variables
        for abbrev in sorted_bands:
            osi_name = band_mapping[abbrev]
            # Use regex to match whole words only (not part of other words)
            # Pattern: \b matches word boundary, but we need to handle cases like "RE1" in "RE1*RE2"
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            result = re.sub(pattern, osi_name, result)
        
        return result
    
    def formula_osi(self, index_name: str) -> str:
        """
        Get the formula for a spectral index with OSI band names.
        
        Parameters
        ----------
        index_name : str
            Name of the spectral index (e.g., 'NDVI', 'EVI', 'NBR')
        
        Returns
        -------
        str
            Mathematical formula string using OSI band names
            (nir, red, green, blue, swir1, swir2, redE1-4)
        
        Examples
        --------
        >>> utils = SpectralIndicesUtils()
        >>> utils.formula_osi('NDVI')
        '(nir - red)/(nir + red)'
        
        >>> utils.formula_osi('NBR')
        '(nir - swir2)/(nir + swir2)'
        """
        formula_orig = self.formula(index_name)
        return self.formula_to_osi_bands(formula_orig)
    
    def list_indices(self, pattern: Optional[str] = None) -> List[str]:
        """
        List available spectral indices, optionally filtered by pattern.
        
        Parameters
        ----------
        pattern : str, optional
            Pattern to filter indices (case-insensitive). If None, returns all.
        
        Returns
        -------
        list
            List of spectral index names
        """
        if pattern is None:
            return self.spectral_indices_list.copy()
        
        pattern_upper = pattern.upper()
        return [
            idx for idx in self.spectral_indices_list 
            if pattern_upper in idx.upper()
        ]


# Convenience functions for direct use (without instantiating class)
_indices_utils_instance = None


def _get_utils_instance() -> SpectralIndicesUtils:
    """Get or create the global SpectralIndicesUtils instance."""
    global _indices_utils_instance
    if _indices_utils_instance is None:
        _indices_utils_instance = SpectralIndicesUtils()
    return _indices_utils_instance


def get_index_info(index_name: str) -> Dict[str, Any]:
    """Get formula and metadata for a spectral index (convenience function)."""
    return _get_utils_instance().get_index_info(index_name)


def formula(index_name: str) -> str:
    """Get the formula for a spectral index (convenience function)."""
    return _get_utils_instance().formula(index_name)


def formula_to_osi_bands(formula_str: str) -> str:
    """Convert eemont-osi formula to OSI band names (convenience function)."""
    return _get_utils_instance().formula_to_osi_bands(formula_str)


def formula_osi(index_name: str) -> str:
    """Get formula with OSI band names (convenience function)."""
    return _get_utils_instance().formula_osi(index_name)


def list_indices(pattern: Optional[str] = None) -> List[str]:
    """List available spectral indices (convenience function)."""
    return _get_utils_instance().list_indices(pattern)


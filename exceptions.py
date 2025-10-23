"""
Custom exceptions for Forestry Carbon ARR library.
"""


class ForestryCarbonError(Exception):
    """Base exception for Forestry Carbon ARR library."""
    pass


class DependencyError(ForestryCarbonError):
    """Raised when required dependencies are missing."""
    pass


class ConfigurationError(ForestryCarbonError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(ForestryCarbonError):
    """Raised when data processing fails."""
    pass


class GEEError(ForestryCarbonError):
    """Raised when Google Earth Engine operations fail."""
    pass


class MLError(ForestryCarbonError):
    """Raised when machine learning operations fail."""
    pass


class SatelliteError(ForestryCarbonError):
    """Raised when satellite data processing fails."""
    pass


class ValidationError(ForestryCarbonError):
    """Raised when data validation fails."""
    pass

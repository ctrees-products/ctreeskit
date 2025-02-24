"""
XR Analyzer - Xarray-based spatial analysis tools
"""

from .xr_spatial_processor import XrSpatialProcessor
from .xr_zonal_stats import XrZonalStats
from .common import Units

__version__ = "0.1.0"
__author__ = "Naomi Provost"
__email__ = "nprovost@ctrees.org"

__all__ = [
    "XrSpatialProcessor",
    "XrZonalStats",
    "Units"
]

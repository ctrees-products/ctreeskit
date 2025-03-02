"""
XR Analyzer - Xarray-based spatial analysis tools
"""

from .xr_spatial_processor import XrSpatialProcessor
from .xr_zonal_stats import XrZonalStats
from .xr_geometry_processor import XrGeometryProcessor
from .common import Units

__version__ = "0.1.0"
__author__ = "Naomi Provost"
__email__ = "nprovost@ctrees.org"

__all__ = [
    "XrSpatialProcessor",
    "XrGeometryProcessor",
    "XrZonalStats",
    "Units"
]

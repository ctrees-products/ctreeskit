"""
XR Analyzer - Xarray-based spatial analysis tools
"""

# Import and expose key functions from the modules
from .xr_spatial_processor_module import (
    process_geometry,
    clip_ds_to_bbox,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    create_proportion_geom_mask,
    align_and_resample_ds,
    GeometryData
)

from .xr_zonal_stats_module import (
    calculate_categorical_area_stats
)

__version__ = "0.1.1"
__author__ = "Naomi Provost"
__email__ = "nprovost@ctrees.org"


__all__ = [
    # From spatial processor
    "process_geometry",
    "clip_ds_to_bbox",
    "clip_ds_to_geom",
    "create_area_ds_from_degrees_ds",
    "create_proportion_geom_mask",
    "align_and_resample_ds",
    "GeometryData",

    # From zonal stats
    "calculate_categorical_area_stats"
]

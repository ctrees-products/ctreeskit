# Import and expose key functions from the modules
from .xr_analyzer.xr_spatial_processor_module import (
    process_geometry,
    clip_ds_to_bbox,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    create_proportion_geom_mask,
    align_and_resample_ds,
    GeometryData
)

from .xr_analyzer.xr_zonal_stats_module import (
    calculate_categorical_area_stats
)
from .arraylake_tools.common import ArraylakeDatasetConfig
from .arraylake_tools.create import ArraylakeRepoCreator
from .arraylake_tools.initialize import ArraylakeRepoInitializer
from .arraylake_tools.populate_dask import ArraylakeRepoPopulator

__version__ = "0.1.1"
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
    "calculate_categorical_area_stats",

    # From arraylake tools
    "ArraylakeDatasetConfig",
    "ArraylakeRepoCreator",
    "ArraylakeRepoInitializer",
    "ArraylakeRepoPopulator"
]

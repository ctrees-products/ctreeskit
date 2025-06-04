# Import and expose key functions from the modules
from .xr_analyzer.xr_spatial_processor_module import (
    process_geometry,
    clip_ds_to_bbox,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    create_proportion_geom_mask,
    reproject_match_ds,
    GeometryData,
)

from .xr_analyzer.xr_common import (
    get_single_var_data_array,
    get_flag_meanings,
    agg_classified_mapped_da,
)

from .xr_analyzer.xr_zonal_stats_module import (
    calculate_categorical_area_stats,
    calculate_combined_categorical_area_stats,
    create_combined_classification,
    calculate_stats_with_categories,
)

from .arraylake_tools.common import ArraylakeDatasetConfig
from .arraylake_tools.create import ArraylakeRepoCreator
from .arraylake_tools.initialize import ArraylakeRepoInitializer
from .arraylake_tools.populate_dask import ArraylakeRepoPopulator

from .dask_analyzer.dask_calc_categorical_area_stats import (
    _prepare_area_ds_dask,
    calculate_categorical_area_stats_dask,
)
from .dask_analyzer.dask_create_area_ds_from_degrees import (
    create_area_ds_from_degrees_ds_dask,
)

from .dask_analyzer.dask_reproj_match import reproject_match_dask

from .dask_analyzer.dask_geometry_clip_rio import geometry_clip_rio


__version__ = "0.1.1"
__all__ = [
    # From spatial processor
    "process_geometry",
    "clip_ds_to_bbox",
    "clip_ds_to_geom",
    "create_area_ds_from_degrees_ds",
    "create_proportion_geom_mask",
    "reproject_match_ds",
    "GeometryData",
    # From xr common
    "get_single_var_data_array",
    "get_flag_meanings",
    "agg_classified_mapped_da",
    # From zonal stats
    "calculate_categorical_area_stats",
    "calculate_combined_categorical_area_stats",
    "create_combined_classification",
    "calculate_stats_with_categories",
    # From arraylake tools
    "ArraylakeDatasetConfig",
    "ArraylakeRepoCreator",
    "ArraylakeRepoInitializer",
    "ArraylakeRepoPopulator",
    ## from dask analyzer
    "_prepare_area_ds_dask",
    "create_area_ds_from_degrees_ds_dask",
    "calculate_categorical_area_stats_dask",
    "geometry_clip_rio",
    "reproject_match_dask",
]

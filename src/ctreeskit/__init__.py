from importlib.metadata import PackageNotFoundError, version

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
from .arraylake_tools.ingest import AnnualRasterIngester

# Single-sourced from pyproject.toml's [project].version via package metadata.
try:
    __version__ = version("ctreeskit")
except PackageNotFoundError:  # running from a checkout without installation
    __version__ = "0.0.0+unknown"
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
    "AnnualRasterIngester",
]

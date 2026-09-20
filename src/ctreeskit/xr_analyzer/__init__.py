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
    reproject_match_ds,
    GeometryData,
)

from .xr_common import (
    get_single_var_data_array,
    get_flag_meanings,
    agg_classified_mapped_da,
)

from .xr_zonal_stats_module import (
    calculate_categorical_area_stats,
    calculate_combined_categorical_area_stats,
    create_combined_classification,
    calculate_stats_with_categories,
)

from .xr_vectorize_module import (
    patches_from_mask,
    patches_from_categorical,
    patches_over_time,
    merge_patches,
    assign_to_polygons,
    write_geoparquet,
    read_geoparquet,
)

from .xr_observations_module import (
    STATE_CODES,
    STATE_FILL,
    DATE_PRECISION,
    DEFAULT_CODE_OFFSETS,
    ObservationField,
    OBSERVATION_FIELDS,
    OBSERVATION_ATTRS,
    OBSERVATION_TABLE_DTYPES,
    observations_from_dated_codes,
    observations_from_annual_alert_days,
    select_month,
    observations_from_tier_steps,
    observations_from_tier_steps_by_step,
    select_step_month,
    observations_from_points,
    points_to_grid_mask,
    observations_to_table,
    write_observations,
    read_observations,
    observations_mask,
)

__all__ = [
    # From spatial processor
    "process_geometry",
    "clip_ds_to_bbox",
    "clip_ds_to_geom",
    "create_area_ds_from_degrees_ds",
    "create_proportion_geom_mask",
    "reproject_match_ds",
    "GeometryData",
    # From xr_common
    "get_flag_meanings",
    "get_single_var_data_array",
    "agg_classified_mapped_da",
    # From zonal stats
    "calculate_categorical_area_stats",
    "calculate_combined_categorical_area_stats",
    "create_combined_classification",
    "calculate_stats_with_categories",
    # From vectorize
    "patches_from_mask",
    "patches_from_categorical",
    "patches_over_time",
    "merge_patches",
    "assign_to_polygons",
    "write_geoparquet",
    "read_geoparquet",
    # From observations
    "STATE_CODES",
    "STATE_FILL",
    "DATE_PRECISION",
    "DEFAULT_CODE_OFFSETS",
    "ObservationField",
    "OBSERVATION_FIELDS",
    "OBSERVATION_ATTRS",
    "OBSERVATION_TABLE_DTYPES",
    "observations_from_dated_codes",
    "observations_from_annual_alert_days",
    "select_month",
    "observations_from_tier_steps",
    "observations_from_tier_steps_by_step",
    "select_step_month",
    "observations_from_points",
    "points_to_grid_mask",
    "observations_to_table",
    "write_observations",
    "read_observations",
    "observations_mask",
]

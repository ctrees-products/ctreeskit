# `xr_analyzer` Module

This module provides tools for analyzing and processing spatial data using `xarray`. Below are the instructions on how to use the functions provided in the module.


## Quick Links
- [xr_spatial_processor_module](#xr_spatial_processor_module)
    - [Functions](#xr_spatial_processor-functions)
    - [Notes](#xr_spatial_processor-notes)
- [xr_zonal_stats_module](#xr_zonal_stats_module)
    - [Functions](#xr_spatial_processor-functions)
    - [Notes](#xr_zonal_stats-notes)
- [xr_common_module](#xr_common-module)
    - [Functions](#xr_common-functions)
- [Usage Examples](#usage-examples)

# xr_spatial_processor_module

## Overview

This module provides tools to:

- Process geospatial vector data (from files or objects) into standardized geometry containers
- Clip rasters to geometries or bounding boxes
- Align and resample rasters to match reference grids
- Calculate accurate cell areas for geographic rasters
- Create weighted masks based on geometry-pixel intersections

## xr_spatial_processor Functions
- [xr_spatial_processor_module](#xr_spatial_processor_module)
    - [Notes](#xr_spatial_processor-notes)
    - [GeometryData](#geometrydata-container)
    - [process_geometry](#process_geometry)
    - [clip_ds_to_bbox](#clip_ds_to_bbox)
    - [clip_ds_to_geom](#clip_ds_to_geom)
    - [reproject_match_ds](#reproject_match_ds)
    - [create_proportion_geom_mask](#create_proportion_geom_mask)
    - [create_area_ds_from_degrees_ds](#create_area_ds_from_degrees_ds)

### GeometryData Container

The module uses a `GeometryData` class as a container for processed geometry information:

```python
class GeometryData:
        """Container for spatial geometry information."""
        geom: Optional[List[GeometryLike]]  # List of geometry objects
        geom_crs: Optional[str]            # Coordinate reference system
        geom_bbox: Optional[tuple]         # Bounding box (minx, miny, maxx, maxy)
        geom_area: Optional[float]         # Area (in m² or ha)
```

### `process_geometry`

Load, validate, and process a geometry source into a standardized GeometryData object.



**Parameters:**
- `geom_source` (str or GeometryLike or list of GeometryLike): The input geometry source.
- `dissolve` (bool, default True): If True, all geometries are dissolved into a single geometry.
- `output_in_ha` (bool, default True): If True, converts the computed area from square meters to hectares.

**Example Usage:**

- `out_geom = process_geometry('input.geosjon', dissolve = True, output_in_ha = True)`

**Returns:**
- `GeometryData`: An object containing geometry information.



### `clip_ds_to_bbox`

Clip a raster (DataArray or Dataset) to a given bounding box.

**Parameters:**
- `input_ds` (xr.DataArray or xr.Dataset): The input raster with valid spatial metadata.
- `bbox` (tuple): Bounding box as (minx, miny, maxx, maxy).
- `drop_time` (bool, default False): If True and the raster has a 'time' dimension, only the first time slice is returned.

**Example Usage:**

- `bbox_ds = clip_ds_to_bbox(input_ds = test_ds, bbox=out_geom.geom_bbox)`

**Returns:**
- `xr.DataArray`: (or `xr.DataSet`) The raster clipped to the specified bounding box.

### `clip_ds_to_geom`

Clip a raster to the extent of the provided geometry.

**Parameters:**
- `input_ds` (xr.DataArray or xr.Dataset): The input raster to be clipped. Must contain spatial metadata.
- `geom_source` (ExtendedGeometryInput): Either a GeometryData instance, a single geometry (or list), or a GeoJSON file path.
- `all_touch` (bool, default False): If True, includes all pixels touched by the geometry boundaries.

**Example Usage:**

- `clipped_ds = clip_ds_to_geom(input_ds = test_ds, geom_source=out_geom.geom)`

**Returns:**
- `xr.DataArray`: (or `xr.DataSet`) : Raster clipped to the geometry’s spatial extent.

### `reproject_match_ds`

Align and resample a target raster to match the spatial grid of a template raster.

**Parameters:**
- `template_raster` (xr.DataArray or xr.Dataset): The reference raster defining the target grid.
- `target_raster` (xr.DataArray or xr.Dataset): The raster to be aligned and resampled.
- `resampling_method` (str, optional): The resampling algorithm to use.
- `return_area_grid` (bool, default True): If True, returns a DataArray with grid cell areas.
- `output_in_ha` (bool, default True): If True, computed areas will be converted to hectares.

**Example Usage:**

- `mod_ds, area_ds = reproject_match_ds(template_raster = clipped_ds, target_raster=new_ds)`

**Returns:**
- `tuple`: A tuple (aligned_target, area_target).

### `create_proportion_geom_mask`

Create a weighted mask for a raster based on the intersection proportions of its pixels with a geometry.

**Parameters:**
- `input_ds` (xr.DataArray): The input raster whose pixel intersection proportions are to be computed.
- `geom_source` (ExtendedGeometryInput): Either a GeometryData instance or a raw geometry source.
- `pixel_ratio` (float, default 0.001): The minimum ratio of pixel area to geometry area required before performing a weighted computation.
- `overwrite` (bool, default False): If True, bypasses the pixel_ratio check and always computes weighted proportions.

**Example Usage:**

- `porportion_ds = create_proportion_geom_mask(input_ds, out_geom.geom, overwrite=True)`

**Returns:**
- `xr.DataArray`: A DataArray mask where each pixel value represents the fraction of that pixel's area that intersects the geometry.

### `create_area_ds_from_degrees_ds`

Create an area DataArray from a dataset with degree coordinates.

**Parameters:**
- `input_ds` (xr.DataArray or xr.Dataset): Input dataset with degree coordinates.
- `high_accuracy` (Optional[bool], default None): If True, uses high accuracy calculations.
- `output_in_ha` (bool, default True): If True, computed areas will be converted to hectares.

**Example Usage:**

- `area_ds = create_area_ds_from_degrees_ds(input_ds)`

**Returns:**
- `xr.DataArray`: DataArray with area values.

## xr_spatial_processor Notes

- By default, area calculations use a heuristic based on latitude: geodesic calculations for high latitudes (above 70°) and equal-area projection (EPSG:6933) otherwise.
- When creating proportion masks, the module checks if pixel sizes are too small relative to the geometry; this can be overridden with `overwrite=True`.
- For clipping operations, the module supports both the "all_touch" (include pixels touching the boundary) and standard intersection modes.
- For time-series data, `clip_ds_to_bbox` can optionally drop the time dimension with `drop_time=True`.



# xr_zonal_stats_module

## Overview

This module provides tools to:

- Calculate area statistics for different classes in categorical rasters.
- Support both time-series and static (non-temporal) raster data.
- Offer flexible area calculation options (pixel counts, constant values, or spatially-variable areas).
- Generate tabular summaries as pandas DataFrames.

## xr_zonal_stats Functions
- [xr_zonal_stats_module](#xr_zonal_stats_module)
    - [Notes](#xr_zonal_stats-notes)
    - [calculate_categorical_area_stats](#calculate_categorical_area_stats)

### `calculate_categorical_area_stats`

Calculate area statistics for each class in categorical raster data.

**Parameters:**
- `categorical_ds` (xr.Dataset or xr.DataArray): Categorical raster data (with or without time dimension).
- `area_ds` (None, bool, float, or xr.DataArray, optional): Area per pixel.
- `var_name` (str, default None): Name of the variable in the dataset containing class values.
- `count_name` (str, default "area_hectares"): Name for the metric column in the output DataFrame.
- `reshape` (bool, default True): If True, pivots output to wide format with classes as columns.
- `drop_zero` (bool, default True): If True, removes class 0 (typically no-data) from results.

**Example Usage:**

- `area_stats_df = calculate_categorical_area_stats(input_ds, area_ds=True)`

**Notes:**
- The function treats 0 values specially, ensuring they remain 0 (useful for nodata values)
- When `area_ds=True`, the module uses `create_area_ds_from_degrees_ds()` from the spatial processor module
- For datasets with flag metadata, class columns will be renamed using flag meanings
- For datasets with many classes, consider using drop_zero=True to exclude no-data values from results

**Returns:**
- `pd.DataFrame`: Results with columns: class values as columns and "total_area".

### `calculate_combined_categorical_area_stats`

Calculate area statistics for unique combinations of two categorical datasets.

**Parameters:**
- `primary_ds` (xr.DataArray): First categorical raster dataset.
- `secondary_ds` (xr.DataArray): Second categorical raster dataset.
- `area_ds` (None, bool, float, or xr.DataArray, optional): Area per pixel.
- `count_name` (str, default "area_hectares"): Name for the metric column in the output DataFrame.
- `reshape` (bool, default True): If True, pivots output to wide format with classes as columns.
- `drop_zero` (bool, default True): If True, removes combinations where either dataset has a value of 0.

**Example Usage:**

- `combined_stats_df = calculate_combined_categorical_area_stats(primary_ds, secondary_ds, area_ds=True)`

**Returns:**
- `pd.DataFrame`: Results with columns: original classifications, their flags, and total area.

**Notes:**
- The function ensures both datasets are aligned spatially using `reproject_match_ds`.
- Class combinations are represented as strings (e.g., "1.2" for class 1 in the primary dataset and class 2 in the secondary dataset).
- For datasets with flag metadata, class columns will be renamed using flag meanings from both datasets.
- Use `drop_zero=True` to exclude combinations involving no-data values.
- When `area_ds=True`, the module uses `create_area_ds_from_degrees_ds()` from the spatial processor module.
- For large datasets, consider optimizing memory usage by processing in chunks.


## xr_zonal_stats Notes
- Ensure that input datasets have consistent dimensions and coordinate systems for accurate results.
- Ensure that dataset has attribute "classification"
- The module assumes that class value `0` typically represents no-data and provides an option to exclude it from the results.
- Metadata such as flag meanings can be used to enhance the interpretability of the output.


# xr_common module
## Overview

This module provides tools to:

- Get single data arary from a dataset
- Return values of "Flag Meanings" if using flag cf convention

## xr_common Functions
- [xr_common module](#xr_common-module)
    - [get_single_var_data_array](#get_single_var_data_array)
    - [get_flag_meanings](#get_flag_meanings)
    
### `get_single_var_data_array`

Get the single DataArray from the input dataset.

**Parameters:**
- `xr_dataset` (xr.Dataset or xr.DataArray): Input dataset.
- `var_name` (str): Name of the variable to extract.

**Example Usage:**

- `single_data_array = get_single_var_data_array(input_ds,"classification")`

**Returns:**
- `xr.DataArray`: Extracted DataArray.

### `get_flag_meanings`

Get flag meanings from the dataset attributes.

**Parameters:**
- `xr_dataset` (xr.Dataset): Input dataset.

**Example Usage:**

- `flag_meanings = get_flag_meanings(input_ds)`

**Returns:**
- `list`: List of flag meanings.


# Usage Examples
```python
import xarray as xr
from ctreeskit import (
    process_geometry, clip_ds_to_bbox, clip_ds_to_geom,
    reproject_match_ds, create_proportion_geom_mask,
    create_area_ds_from_degrees_ds, calculate_categorical_area_stats
)

# Example: Clip dataset to bounding box
ds = xr.open_dataset('path_to_raster_data.tif')
bbox = (-120.0, 35.0, -119.0, 36.0)
clipped_ds = clip_ds_to_bbox(ds, bbox)

# Example: Clip dataset to geometry
geom_data = process_geometry('path_to_geojson_file.geojson')
clipped_geom_ds = clip_ds_to_geom(ds, geom_data)

# Example: Align and resample dataset
template_ds = xr.open_dataset('path_to_template_raster.tif')
aligned_ds, area_grid = reproject_match_ds(template_ds, ds)

# Example: Create proportion geometry mask
geom_data = process_geometry('path_to_geojson_file.geojson')
proportion_mask = create_proportion_geom_mask(ds, geom_data)

# Example: Create area dataset from degrees dataset
area_ds = create_area_ds_from_degrees_ds(ds)

area_stats = calculate_categorical_area_stats(ds, area_ds)

area_stats.to_csv("output_area_stats.csv")
```

# `xr_analyzer` Module

This module provides tools for analyzing and processing spatial data using `xarray`. Below are the instructions on how to use the functions provided in the module.


## Quick Links
- [xr_spatial_processor_module](#xr_spatial_processor_module)
    - [Functions](#xr_spatial_processor-functions)
    - [Notes](#xr_spatial_processor-notes)
- [xr_zonal_stats_module](#xr_zonal_stats_module)
    - [Functions](#xr_spatial_processor-functions)
    - [Notes](#xr_zonal_stats-notes)
- [xr_vectorize_module](#xr_vectorize_module)
    - [Functions](#xr_vectorize-functions)
    - [Notes](#xr_vectorize-notes)
- [xr_observations_module](#xr_observations_module)
    - [Functions](#xr_observations-functions)
    - [Notes](#xr_observations-notes)
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


# xr_vectorize_module

## Overview

Turns categorical change layers into vector patches and events. Vector output
needs the optional `vector` extra (`pip install 'ctreeskit[vector]'`, which
brings geopandas and pyarrow); the module imports without it and the functions
that need it raise `ImportError` with install instructions.

## xr_vectorize Functions

### `patches_from_mask`

Polygonize the connected regions of a 2-D boolean or integer raster.

**Parameters:**
- `mask` (xr.DataArray): 2-D raster with a rioxarray CRS and transform.
- `connectivity` (int, default 8): 4 (edge-sharing) or 8 (edges and corners).
- `min_area_ha` (float, optional): minimum mapping unit, applied to measured area.
- `fill_value` (optional): value marking nodata, treated as background.
- `attrs` (dict, optional): constant columns broadcast onto every row.
- `area_ha_grid` (ndarray, optional): per-pixel hectares for the grid, shaped
  like the mask. Measured from the CRS and transform when omitted; pass it to
  reuse one measurement across layers on the same grid.

**Returns:** a `GeoDataFrame` with `patch_id`, `pixel_count`, `area_ha`,
`geometry` and any `attrs` keys, carrying the raster's CRS.

### `patches_from_categorical`

Builds the mask from `da.isin(values)` and calls `patches_from_mask`, recording
the matched value(s) in a `class_values` column.

### `patches_over_time`

Iterates a time dimension (`time_dim`, default `"time"`), polygonizing one step
at a time and concatenating the result with a `time` column (a
`pandas.Timestamp` for datetime coordinates). The per-pixel area grid is
measured once and shared by every step.

### `merge_patches`

Groups patches within `gap_m` metres of each other into events, optionally only
within matching `group_by` columns.

**Returns:** `(events, membership)`. `events` carries `event_id`, `area_ha`,
`patch_count`, `pixel_count`, the `group_by` columns and the dissolved member
geometry; `membership` is an `event_id, patch_id` table. `area_ha` is the sum
of member areas, not a remeasurement, so patches that overlap in space (the
same location in several time steps merged without `group_by=["time"]`) count
once per patch.

### `assign_to_polygons`

Relates events to reference polygons many-to-many.

**Returns:** a DataFrame of `event_id`, `<polygon_id>`, `overlap_area_ha` and
`overlap_frac` (the share of the event's area inside that polygon), with
non-overlapping pairs and pairs below `min_overlap_frac` omitted.

### `write_geoparquet` / `read_geoparquet`

Round-trip a GeoDataFrame as GeoParquet 1.0.0 with WKB geometry. Writing a frame
without a CRS raises, since readers default a missing CRS to `OGC:CRS84`.

## xr_vectorize Notes

- Area is true area in hectares. On geographic grids each cell is measured with
  `create_area_ds_from_degrees_ds` (geodesic, or an EPSG:6933 equal-area
  approximation), so cells shrink poleward. On projected grids the constant
  pixel area comes from the affine transform, and non-metre axis units are
  rejected.
- `merge_patches` buffers in a projected CRS — the estimated UTM zone when the
  input is geographic, or an explicit `metric_crs` — so `gap_m` is always a true
  distance. Event geometries are the union of the member patches, not the
  buffered hulls.
- `assign_to_polygons` measures overlap in EPSG:6933, so the fractions for one
  event sum to 1 when the polygons cover it.


# xr_observations_module

## Overview

Normalizes detection layers from unlike sources into one *observation record*.
An observation is a detection episode at a single pixel (or point) from a single
source: where it is, which source and sensor saw it, when it was first seen,
confirmed and last seen, how many looks the source had and how many were
positive, what state it is in, how confident it is, an optional class hint, how
precise the dates are, and which snapshot it was read from. Pixels that never
detected anything are not observations -- they contribute only the look counts
carried on the detections around them.

Parquet output needs the optional `vector` extra
(`pip install 'ctreeskit[vector]'`, which brings pyarrow); the import is
deferred, so the module imports without it.

## The schema

`OBSERVATION_FIELDS` is the normalized variable list, each entry an
`ObservationField` with a `name`, `dtype`, `fill` and `description`:

| Variable | dtype | Unknown |
|---|---|---|
| `state` | `uint8` | 255 |
| `first_obs_date`, `confirm_date`, `last_obs_date` | `datetime64[ns]` | `NaT` |
| `n_obs`, `n_positive` | `uint16` | 0 |
| `confidence_native`, `confidence` | `float32` | `NaN` |
| `class_hint` | `uint8` | 0 |
| `date_precision` | `uint8` | 0 |

`STATE_CODES`: 0 none, 1 candidate, 2 confirmed, 3 rejected, 4 alert,
5 point_detection; `STATE_FILL` is 255. `DATE_PRECISION`: 0 unknown, 1 day,
2 month, 3 lower_bound. `confidence_native` is the source's own scale;
`confidence` is rescaled to [0, 1].

Every record carries the attributes in `OBSERVATION_ATTRS` -- `source`,
`sensor`, `pixel_size_m`, `snapshot_id`, `read_at` (ISO 8601) and `has_counts`
-- plus its CRS, written by rioxarray. `has_counts` is what says whether `n_obs`
and `n_positive` are real counts or the 0 that means "unknown".

**Grid layout.** A normalized record lives on the **source grid** -- nothing is
resampled -- and is 2-D `(y, x)` per detection-episode layer. Where a source has
a time axis, the adapter either collapses each pixel to its episodes or keeps
the `time` dimension, and says which: episodes that cannot overlap collapse to a
plain 2-D record, and episodes that can (the same pixel detected again in a
later step) keep the dimension, because collapsing them would silently drop all
but one.

## xr_observations Functions

### `observations_from_dated_codes`

Decodes a 2-D integer raster with **no time dimension** whose pixels hold dated
state codes: a state band plus an encoded year/month, for example `2403` for a
confirmed detection in March 2024, `12403` for a candidate in the same month and
`22403` for a rejected one. Stable and fill values decode to state 0. The
output is 2-D, since one such raster carries one episode per pixel.

**Parameters:**
- `da` (xr.DataArray): 2-D integer raster with a rioxarray CRS. Dask-backed
  input is decoded chunk by chunk and stays lazy.
- `code_table` (mapping, optional): state name to output state code; defaults to
  `STATE_CODES`.
- `epoch_format` ({"YYMM", "YYYYMM"}, default "YYMM") and `century` (int,
  default 2000): how the date is encoded inside the band.
- `offsets` (mapping, default `DEFAULT_CODE_OFFSETS`): state name to the
  additive offset marking its band, each a whole multiple of the band width
  (10000 for `YYMM`, 1000000 for `YYYYMM`).
- `stable_values` (sequence of int, default `(0,)`) and `fill_value` (int,
  optional): values that decode to state 0.
- `date_precision` ({"day", "month", "lower_bound"}, default "month").
- `lower_bound_months` (sequence of int, optional): month numbers whose
  detections are recorded as `date_precision = 3` instead.
- `class_hint_map` (mapping, optional): state name to a `class_hint` value.
- `snapshot_id`, `source`, `sensor`, `pixel_size_m`: provenance.

**Returns:** the normalized `Dataset` on the input's `(y, x)` grid.
`last_obs_date` is NaT and `has_counts` is False -- a single dated code says
when a detection happened, not when it was last seen.

### `observations_from_annual_alert_days`

Decodes a pair of rasters with an annual `time` dimension: `alert`, where a
non-zero value marks an alert, and `date`, holding the day of that alert as an
integer offset from `day_epoch`. The `time` dimension is **kept**: a pixel may
alert in more than one year, and those are separate episodes.

**Parameters:** `alert`, `date`, `day_epoch` (default `"1970-01-01"`),
`time_dim` (default `"time"`), `fill_value` (default -9999), `class_hint_map`
(alert value to class hint), and the provenance arguments.

**Returns:** the normalized `Dataset` on the `(time, y, x)` source grid.
`first_obs_date`, `confirm_date` and `last_obs_date` all hold the decoded day,
`date_precision` is 1 (day) and `state` is 4 (alert).

### `select_month`

`select_month(ds, year, month)` returns the boolean mask of the alerts whose
`first_obs_date` falls in that calendar month, shaped like `ds.state`.

### `observations_from_tier_steps`

Collapses a regularly stepped categorical cube -- monthly, annual or any other
fixed period -- in which a detected pixel carries a small **tier code** at one
step and 0 or a fill value at every other. The tier code says how firm the
detection is; the step supplies the date. Different layers spell the same tiers
with different integers, so `tier_map` routes a layer's own vocabulary onto
`STATE_CODES`, given either as names (`{1: "confirmed", 2: "candidate"}`) or as
codes (`{1: 2, 2: 1}`).

The step dimension is **collapsed**: one write per pixel is one episode, so a
plain 2-D record holds it without loss.

**Parameters:**
- `da` (xr.DataArray): stepped categorical raster with a rioxarray CRS.
  Dask-backed input is collapsed chunk by chunk and stays lazy; the step axis is
  rechunked to one chunk, since the reduction reads every step of a pixel.
- `tier_map` (mapping of int to str or int): source value to normalized state.
  Values absent from the mapping, and values mapped to `"none"`, count as
  nothing written, exactly like 0.
- `time_dim` (str, default `"time"`): the step dimension. Its coordinate must be
  date-like and hold the **start** of each period; an integer step index is
  rejected rather than read as an epoch offset.
- `fill_value` (int, optional): value meaning "no data", taking precedence over
  `tier_map` where the two name the same value.
- `date_precision` ({"unknown", "day", "month", "lower_bound"}, default
  "month"): a monthly step names a month, not a day.
- `class_hint` (int, default 0): written wherever a detection was found. One
  layer carries one tier vocabulary, so the hint is a constant.
- `snapshot_id`, `source`, `sensor`, `pixel_size_m`: provenance.

**Returns:** the normalized `Dataset` on the input's `(y, x)` grid, plus a
boolean `multi_step`. `first_obs_date` is the step's coordinate value,
`confirm_date` repeats it where the state is confirmed, `last_obs_date` is NaT
and `has_counts` is False. A pixel written at more than one step keeps its
**latest** write and is flagged `multi_step = True`, which is what makes the
others visible. `multi_step` rides on the Dataset only; it is not a table
column, so the flat schema is unchanged.

### `observations_from_tier_steps_by_step`

The same input and vocabulary handling without the collapse: every step keeps
its own slice, for callers that want per-step masks. Being elementwise, it is
chunk-local along the step axis too. There is no `multi_step` variable --
nothing is collapsed, so no write is hidden.

### `select_step_month`

`select_step_month(ds_by_step, year, month)` returns the boolean mask of the
detections in a per-step record whose `first_obs_date` falls in that calendar
month, shaped like `ds_by_step.state`.

### `observations_from_points`

Normalizes a DataFrame of point detections. Points have no grid, so this adapter
returns the **table** form directly, with `state = 5` and
`date_precision = 1`.

**Parameters:** `df`, `x_col`, `y_col`, `datetime_col`, `confidence_col`
(optional), `confidence_map` (optional, maps a categorical confidence onto
[0, 1]; without it the column must be numeric and values above 1 are read as
percentages), `sensor_col`, `class_col`, `class_hint_map`, `crs` (recorded on
the returned table's `attrs["crs"]`), and the provenance arguments.

### `points_to_grid_mask`

`points_to_grid_mask(table, like, crs=None)` rasterizes a point table onto a
reference grid, reprojecting when the CRSs differ and dropping points outside
the grid.

**Returns:** a boolean `(y, x)` `DataArray` on `like`'s grid, True in cells
holding at least one point -- ready for co-location with a raster record.

### `observations_to_table`

Flattens a record to one row per **detected** pixel; pixels with `state` 0 or
255 are not rows.

**Returns:** a DataFrame with the columns and dtypes of
`OBSERVATION_TABLE_DTYPES`: the schema fields, the pixel centres `x` and `y` in
the dataset's CRS, and `source`, `sensor`, `pixel_size_m` and `snapshot_id` read
from the attributes. `source`, `sensor` and `snapshot_id` are categoricals.

### `write_observations` / `read_observations`

Round-trip an observation table as Parquet through pyarrow, preserving the
categorical columns and NaT. Writing a table missing a schema column raises.

### `observations_mask`

`observations_mask(ds, states=(2,), month=None)` reduces a record to the 2-D
boolean mask that `patches_from_mask` expects, optionally filtered to detections
whose `first_obs_date` falls in a given `(year, month)`. This is the bridge from
observations to patches and events.

## xr_observations Notes

- The adapters never resample. A normalized record sits on the grid it was read
  from, so it stays directly comparable with its source pixels; bring records
  from different grids together through `observations_to_table` or by
  rasterizing one onto the other's grid.
- Every adapter yields the same table columns and dtypes, so records from
  different sources concatenate with `pandas.concat` and the `source` column
  keeps them apart. A Dataset may carry extra variables (`multi_step`); the
  table schema stays fixed.
- `date_precision` is what makes dates from unlike sources comparable. A
  `YYMM`-style code names a month, not a day, so its dates are the first of the
  month at precision 2. A source that cannot observe during part of the year and
  dates everything it missed to the first month it could see again produces
  dates that bound the detection from below rather than locating it: name those
  months in `lower_bound_months` and they are recorded at precision 3 while the
  rest of the year stays at 2.
- A table's `attrs["crs"]` does not survive a Parquet round trip; pass `crs=` to
  `points_to_grid_mask` after reading one back.


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

## Coverage-fraction weighted zonal statistics

For area statistics weighted by *how much* of each pixel a geometry covers
(rather than a binary in/out mask), use
[rasterix](https://rasterix.readthedocs.io/)'s exactextract-backed coverage
(installed with the `zonal` extra: `pip install "ctreeskit[zonal]"`). The
coverage grid is stored sparse — only pixels the geometry touches are
materialized — and the computation is dask-aware, so chunked rasters stay lazy
and no full-size dense mask is ever written out.

```python
import geopandas as gpd
from rasterix.rasterize.exact import coverage

from ctreeskit import calculate_categorical_area_stats, create_area_ds_from_degrees_ds

classes = ...  # categorical (y, x) DataArray with CRS metadata
geoms = gpd.read_file("aoi.geojson")

# fraction of each pixel covered by the geometry: dims (geometry, y, x)
cover = coverage(classes, geoms, coverage_weight="fraction")

# densify one geometry's (sparse) coverage grid and weight per-pixel areas by it
cover_2d = cover.isel(geometry=0)
cover_2d = cover_2d.copy(data=cover_2d.data.todense())
area = create_area_ds_from_degrees_ds(classes)

stats = calculate_categorical_area_stats(classes, area_ds=area * cover_2d)
```

Edge pixels contribute only their covered fraction, so `total_area` matches the
geometry's true footprint instead of over-counting boundary pixels.

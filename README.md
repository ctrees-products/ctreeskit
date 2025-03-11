# CTrees Tools

## (( TO BE IMPLEMENTED )) Open Source Components (ctreeskit-core)
```bash
pip install ctreeskit
```

## Quick Links
- [Installation Guide](#installation)
- [Basic Usage Examples](#usage)
- [API Documentation](#api-reference)
  - [XrSpatialProcessor Documentation](#xrspatialprocessor)
  - [XrZonalStats Documentation](#xrzonalstats)

## Table of Contents
1. [Installation](#installation)
   - [PyPI Installation](#from-pypi-to-do--not-yet-implemented)
   - [GitHub Installation](#from-github)
   - [Development Setup](#development-installation)
   - [Testing](#testing)
   - [Contributing](#contributing)
2. [Features](#features)
3. [Usage](#usage)
4. [API Reference](#api-reference)
   - [XrSpatialProcessor](#xrspatialprocessor)



## Installation

You can install ctreeskit either from PyPI or directly from GitHub:

### From PyPI [TO DO- not yet implemented]
```bash
pip install ctreeskit
```

### From GitHub
```bash
pip install git+https://github.com/ctrees-products/ctreeskit.git@module_base_package
```

### Development Installation
For development, you can install with all dependencies:
```bash
# Clone the repository
git clone https://github.com/ctrees-products/ctreeskit.git
cd ctreeskit
change branch to module_base_package

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install in editable mode with all dependencies
pip install -e ".[all]"
```

## Testing

To run the tests for the ctreeskit package, navigate to the project directory and execute:

```
pytest -m tests
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## Features

- Spatial processing with xarray DataArrays
- Area calculations and geometry data

A Python module for efficient geospatial operations on raster data using xarray, with support for integrating vector geometries and calculating areas.

## Installation

This module is part of the `ctreeskit` package. Install via pip:

```bash
pip install ctreeskit
```

## Dependencies

- xarray
- rioxarray (for spatial operations)
- numpy
- shapely
- pyproj
- s3fs (for Amazon S3 storage access)

# XrSpatialProcessor
## Overview

This module provides tools to:

- Process geospatial vector data (from files or objects) into standardized geometry containers
- Clip rasters to geometries or bounding boxes
- Align and resample rasters to match reference grids
- Calculate accurate cell areas for geographic rasters
- Create weighted masks based on geometry-pixel intersections

## Key Features

## GeometryData Container

The module uses a `GeometryData` class as a container for processed geometry information:

```python
class GeometryData:
    """Container for spatial geometry information."""
    geom: Optional[List[GeometryLike]]  # List of geometry objects
    geom_crs: Optional[str]            # Coordinate reference system
    geom_bbox: Optional[tuple]         # Bounding box (minx, miny, maxx, maxy)
    geom_area: Optional[float]         # Area (in m² or ha)
```

The `process_geometry` function returns an instance of this class, which can then be directly passed to other functions in the module:

```python
# Process a geometry source
geom_data = xspm.process_geometry("path/to/geometry.geojson")

# Use the processed geometry directly with other functions
clipped = xspm.clip_ds_to_geom(geom_data, my_raster)
mask = xspm.create_proportion_geom_mask(geom_data, my_raster)

# Access geometry properties
print(f"Geometry area: {geom_data.geom_area} hectares")
print(f"Bounding box: {geom_data.geom_bbox}")
```

This design simplifies workflows by centralizing geometry processing and maintaining consistent behavior across operations.


### Geometry Processing

The module accepts various geometry input formats:
- GeoJSON files (local or S3)
- Shapely geometry objects
- Lists of geometries
- Pre-processed GeometryData objects

```python
import ctreeskit.xr_analyzer.xr_spatial_processor_module as xspm

# From a local GeoJSON file
geom_data = xspm.process_geometry("path/to/geometry.geojson")

# From an S3 path
geom_data = xspm.process_geometry("s3://bucket-name/path/to/geometry.geojson")

# From a shapely geometry
from shapely.geometry import Polygon
poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
geom_data = xspm.process_geometry(poly)
```

### Raster Clipping

Clip xarray DataArrays or Datasets to geometries:

```python
# Clip to a bounding box
clipped_raster = xspm.clip_ds_to_bbox(raster_data, bbox=(minx, miny, maxx, maxy))
# With time handling
clipped_raster = xspm.clip_ds_to_bbox(raster_data, bbox=(minx, miny, maxx, maxy), drop_time=True)

# Clip to a geometry
clipped_raster = xspm.clip_ds_to_geom(raster_data, geom_data)
# With all pixels that touch the boundary
clipped_raster = xspm.clip_ds_to_geom(raster_data, geom_data, all_touch=True)
```

### Area Calculation

Calculate accurate grid cell areas for geographic rasters:

```python
# Calculate areas in hectares (default)
area_grid = xspm.create_area_ds_from_degrees_ds(raster_data)

# Calculate areas in square meters
area_grid = xspm.create_area_ds_from_degrees_ds(raster_data, output_in_ha=False)

# Force high accuracy calculation (using geodesic distances)
area_grid = xspm.create_area_ds_from_degrees_ds(raster_data, high_accuracy=True)
```

### Raster Alignment and Resampling

Align a raster to match another reference raster's grid:

```python
aligned_raster, area_grid = xspm.align_and_resample_ds(
    template_raster, 
    target_raster, 
    resampling_method=Resampling.nearest
)

# Without area grid
aligned_raster, _ = xspm.align_and_resample_ds(
    template_raster,
    target_raster,
    return_area_grid=False
)
```

### Weighted Masks

Create intersection proportion masks:

```python
# Create a weighted mask where values represent intersection proportions
proportion_mask = xspm.create_proportion_geom_mask(raster_data, geom_data)

# Force proportion calculation even for small pixels
proportion_mask = xspm.create_proportion_geom_mask(
    raster_data, 
    geom_data, 
    overwrite=True
)

# Control the pixel ratio threshold
proportion_mask = xspm.create_proportion_geom_mask(
    raster_data,
    geom_data,
    pixel_ratio=0.01,  # 1% threshold,
    overwrite=True
)
```

## API Reference

### Classes

- `GeometryData` - Container for processed geometry information (geometries, CRS, bbox, area)

### Core Functions

- `process_geometry(geom_source, dissolve=True, output_in_ha=True)` - Process raw geometry inputs into GeometryData objects
- `clip_ds_to_bbox(input_ds, bbox, drop_time=False)` - Clip a raster to a bounding box
- `clip_ds_to_geom(input_ds, geom_source, all_touch=False)` - Clip a raster to a geometry
- `create_area_ds_from_degrees_ds(input_ds, high_accuracy=None, output_in_ha=True)` - Calculate grid cell areas for geographic rasters
- `create_proportion_geom_mask(input_ds, geom_source, pixel_ratio=0.001, overwrite=False)` - Create weighted masks based on pixel-geometry intersection
- `align_and_resample_ds(template_raster, target_raster, resampling_method=Resampling.nearest, return_area_grid=True, output_in_ha=True)` - Align and resample rasters to match reference grids

###  Helper Functions
- `_calculate_geometry_area(geom, geom_crs, target_epsg=6933)` - Calculate geometry area in square meters
- `_measure(lat1, lon1, lat2, lon2)` - Calculate geodesic distance between two points

## Notes

- By default, area calculations use a heuristic based on latitude: geodesic calculations for high latitudes (above 70°) and equal-area projection (EPSG:6933) otherwise
- When creating proportion masks, the module checks if pixel sizes are too small relative to the geometry; this can be overridden with `overwrite=True`
- For clipping operations, the module supports both the "all_touch" (include pixels touching the boundary) and standard intersection modes
- For time-series data, `clip_ds_to_bbox` can optionally drop the time dimension with `drop_time=True`

## Examples

See the module docstrings for detailed usage examples for each function.

# XrZonalStats
## Overview

This module provides tools to:

- Calculate area statistics for different classes in categorical rasters
- Support both time-series and static (non-temporal) raster data
- Offer flexible area calculation options (pixel counts, constant values, or spatially-variable areas)
- Generate tabular summaries as pandas DataFrames

## Key Features

## Categorical Area Statistics
Calculate area statistics for different land cover classes or other categorical data:
```python
import xarray as xr
from ctreeskit.xr_analyzer import calculate_categorical_area_stats

# Simple pixel counting
result = calculate_categorical_area_stats(land_cover_raster)

# Using a specific area per pixel (e.g., 30x30m = 900 sq meters)
result = calculate_categorical_area_stats(land_cover_raster, area_ds=900)

# Automatically calculate area from coordinates
result = calculate_categorical_area_stats(land_cover_raster, area_ds=True)

# Only analyze specific classes
result = calculate_categorical_area_stats(
    land_cover_raster, 
    classification_values=[1, 2, 3, 4]
)
```

## Time Series Support
Works with both static and time-series data:
```python 
# Static raster - returns a single-row DataFrame
static_result = calculate_categorical_area_stats(land_cover)

# Time-series raster - returns a multi-row DataFrame with a "time" column
timeseries_result = calculate_categorical_area_stats(land_cover_timeseries)
```
## Results Format
Results are returned as pandas DataFrames:

```python
# Output DataFrame columns:
# - "0": Total area across all classes
# - "1"..."n": Area for each class (columns match classification_values positions)
# - "time": For time-series data

# Example access:
total_area = result["0"]
forest_area = result["1"]  # If forest is the first class value

```

## API Reference

### Core Functions
- `calculate_categorical_area_stats(categorical_ds, area_ds=None, classification_values=None)` - Calculate area statistics for different classes in a raster

### Parameters
- `categorical_ds` : xr.Dataset or xr.DataArray
  - Categorical raster data (with or without time dimension)
  - If Dataset, it's converted to DataArray
- `area_ds` : None, bool, float, or xr.DataArray, optional
  - None: count pixels (area=1.0 per pixel)
  - float/int: constant area per pixel
  - True: calculate area from coordinates
  - DataArray: custom area per pixel
- `classification_values` : list, optional
  - List of class values to analyze
  - Default uses unique values from data

### Returns

- `pd.DataFrame`
  - Results with columns:
    - "0": total area
    - "1"..."n": per-class areas
    - "time": if input has time dimension

### Helper Functions
- `_calculate_area_stats(cl_values, da_class_t, area_ds)` - Calculate area statistics for a single time slice

### Notes
- The function treats 0 values specially, ensuring they remain 0 (useful for nodata values)
- When `area_ds=True`, the module uses create_area_ds_from_degrees_ds() from the spatial processor module
- For datasets with very large number of classes, consider explicitly providing classification_values with classes of interest

### Examples
#### Basic Usage
```python
import xarray as xr
from ctreeskit.xr_analyzer import calculate_categorical_area_stats

# Load a land cover classification
land_cover = xr.open_rasterio("landcover.tif")

# Calculate area statistics (defaults to pixel counting)
results = calculate_categorical_area_stats(land_cover)
print(results)
```
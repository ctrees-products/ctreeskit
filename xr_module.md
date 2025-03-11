# ctreeskit.xr_analyzer.xr_spatial_processor_module

A Python module for efficient geospatial operations on raster data using xarray, with support for integrating vector geometries and calculating areas.

## Overview

This module provides tools to:

- Process geospatial vector data (from files or objects) into standardized geometry containers
- Clip rasters to geometries or bounding boxes
- Align and resample rasters to match reference grids
- Calculate accurate cell areas for geographic rasters
- Create weighted masks based on geometry-pixel intersections

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

# Clip to a geometry
clipped_raster = xspm.clip_ds_to_geom(raster_data, geometry_data)
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
    resampling_method="nearest"
)
```

### Weighted Masks

Create intersection proportion masks:

```python
# Create a weighted mask where values represent intersection proportions
proportion_mask = xspm.create_proportion_geom_mask(raster_data, geometry_data)

# Force proportion calculation even for small pixels
proportion_mask = xspm.create_proportion_geom_mask(
    raster_data, 
    geometry_data, 
    overwrite=True
)
```

## API Reference

### Classes

- `GeometryData` - Container for processed geometry information (geometries, CRS, bbox, area)

### Core Functions

- `process_geometry()` - Process raw geometry inputs into GeometryData objects
- `clip_ds_to_bbox()` - Clip a raster to a bounding box
- `clip_ds_to_geom()` - Clip a raster to a geometry
- `create_area_ds_from_degrees_ds()` - Calculate grid cell areas (meters) for geographic rasters
- `create_proportion_geom_mask()` - Create weighted masks based on pixel-geometry intersection
- `align_and_resample_ds()` - Align and resample rasters to match reference grids

## Notes

- By default, area calculations use a heuristic based on latitude: geodesic calculations for high latitudes (above 70°) and equal-area projection (EPSG:6933) otherwise
- When creating proportion masks, the module checks if pixel sizes are too small relative to the geometry; this can be overridden with `overwrite=True`
- For clipping operations, the module supports both the "all_touch" (include pixels touching the boundary) and standard intersection modes

## Examples

See the module docstrings for detailed usage examples for each function.
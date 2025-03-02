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
     - [`__init__`](#init)
     - [`reproject_match_target_da`](#reproject_match_target_datarget_da)
     - [`create_binary_geom_mask_da`](#create_binary_geom_mask_da)
     - [`create_weighted_geom_mask_da`](#create_weighted_geom_mask_da)
     - [`create_area_mask_da`](#create_area_mask_dainput_projection4236)
     - [`create_weighted_area_geom_mask_da`](#create_weighted_area_geom_mask_dainput_projection4236)
     - [`create_clipped_da_vector`](#create_clipped_da_vector)
     - [`create_clipped_da_raster`](#create_clipped_da_raster)
   - [XrZonalStats](#xrzonalstats)
     - [`__init__`](#init)
     - [`calculate_categorical_stats`](#calculate_categorical_stats)
     - [`calculate_continuous_stats`](#calculate_continuous_statsscaling_factornone)
     - [`calculate_agb_stats`](#calculate_agb_statsscaling_factor10)
     - [`calculate_stats_by_category`](#calculate_stats_by_categoryscaling_factornone-agbtrue)
     - [`calculate_percentage_area_stats`](#calculate_percentage_area_stats)
5. [License](#license)

## Installation

You can install ctreeskit either from PyPI or directly from GitHub:

### From PyPI [TO DO- not yet implemented]
```bash
pip install ctreeskit
```

### From GitHub
```bash
pip install git+https://github.com/ctrees-products/ctreeskit.git@mask_generator
```

### Development Installation
For development, you can install with all dependencies:
```bash
# Clone the repository
git clone https://github.com/ctrees-products/ctreeskit.git
cd ctreeskit

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
- Area calculations and geometry intersections
- Zonal statistics for categorical and continuous data

## Usage

```python
from ctreeskit import XrSpatialProcessor, XrZonalStats

# Initialize processor
processor = XrSpatialProcessor(dataset, geometry)

# Calculate areas
areas = processor.create_area_ha_da()
```


### API Reference
# XrGeometryProcessor

A class for processing geometries with xarray DataArrays, focusing on mask generation and area calculations.

## Key Methods
- `clip_raster_to_geom`: Clip raster using geometry
- `create_binary_geom_mask`: Create binary (0/1) mask
- `create_proportion_geom_mask`: Calculate intersection proportions
- `create_area_geom_mask`: Calculate geometry-weighted areas
- `create_pixel_areas`: Calculate grid cell areas
- `subset_to_bbox`: Subset to geometry's bounding box

## Key Features
- Flexible geometry handling (single or multiple)
- Area calculations with unit conversion
- Intersection proportion calculations
- Memory-efficient processing
- Optional Dask integration for large datasets

## Basic Usage

```python
from ctreeskit import XrGeometryProcessor

# Initialize with geometry
processor = XrGeometryProcessor(
    geom_source="area.geojson",
    dissolve=True
)

# Create masks
binary_mask = processor.create_binary_geom_mask(raster)
weighted_mask = processor.create_proportion_geom_mask(raster)

# Calculate areas
pixel_areas = processor.create_pixel_areas(raster, unit=Units.HA)
weighted_areas = processor.create_area_geom_mask(raster, binary=False)
```

## Class Methods

### `__init__(geom_source, dissolve=True)`

Initialize GeometryProcessor with input geometry.

**Parameters:**
- `geom_source` (Union[str, gpd.GeoDataFrame]): Input geometry as either:
  - Path to GeoJSON/GPKG file (str)
  - GeoDataFrame (gpd.GeoDataFrame)
- `dissolve` (bool): Whether to dissolve all geometries into one (default: True)

**Attributes:**
- `geom` (List[shapely.Geometry]): List of geometries
- `geom_crs` (CRS): Coordinate reference system
- `geom_bbox` (tuple): Bounds (minx, miny, maxx, maxy)
- `geom_area` (float): Total area in square meters
- `geom_mask` (xr.DataArray): Current mask (if generated)
- `mask_type` (MaskType): Type of current mask

### `create_binary_geom_mask(raster)`

Create binary mask from geometry.

**Parameters:**
- `raster` (xr.DataArray): Reference raster for output grid

**Returns:**
- xr.DataArray: Binary mask (1=inside geometry, 0=outside)

### `create_proportion_geom_mask(raster, pixel_ratio=0.001, overwrite=False)`

Calculate intersection proportions between geometry and pixels.

**Parameters:**
- `raster` (xr.DataArray): Reference raster for output resolution and extent
- `pixel_ratio` (float): Minimum pixel/geometry area ratio (default: 0.001)
  - Represents minimum allowed ratio of pixel area to total geometry area
  - Default threshold is 0.1% (0.001) of geometry area
  - Ratios below this threshold trigger a warning and fallback to binary mask
- `overwrite` (bool): Skip ratio check if True (default: False)
  - When False: Enforces pixel_ratio check and may fallback to binary mask
  - When True: Calculates proportions regardless of pixel/geometry size ratio
  - Use True when working with very large geometries or fine resolution rasters

**Returns:**
- xr.DataArray: Intersection proportions (0-1)
  - 1.0: Pixel fully within geometry
  - 0.0: Pixel outside geometry
  - 0.0-1.0: Proportion of pixel intersecting geometry

**Notes:**
- Performance Warning: Computing exact intersection proportions is computationally intensive
- Memory Warning: Large rasters with small pixels relative to geometry may require significant memory
- Fallback Behavior: When pixel_ratio check fails:
  1. Warning is issued showing actual ratio vs threshold
  2. Binary mask is used instead of proportional mask
  3. self.mask_type is set to BINARY
- Use `overwrite=True` to force proportion calculation regardless of ratio

**Returns:**
- xr.DataArray: Intersection proportions (0-1)

### `create_area_geom_mask(raster, binary=True, unit=Units.M2, input_projection=4236)`

Calculate areas for each cell within geometry.

**Parameters:**
- `raster` (xr.DataArray): Input raster
- `binary` (bool): Use binary or weighted mask
- `unit` (Units): Output unit (m², ha, km²)
- `input_projection` (int): EPSG code for calculations

**Returns:**
- xr.DataArray: Cell areas in specified unit

### `create_pixel_areas(raster, unit=Units.M2, input_projection=4236)`

Calculate area of each grid cell (create clipped to geometry)

**Parameters:**
- `raster` (xr.DataArray): Input raster
- `unit` (Units): Output unit
- `input_projection` (int): EPSG code

**Returns:**
- xr.DataArray: Grid cell areas


# XrZonalStats

A class for calculating zonal statistics for categorical and continuous data with flexible area handling.


## Key Methods
- `calculate_categorical_stats`: Calculate areas and counts by category
- `calculate_continuous_stats`: Compute basic statistics (mean, std, etc.)
- `calculate_agb_stats`: Calculate AGB and carbon statistics
- `calculate_stats_by_category`: Combine category and value statistics
- `calculate_percentage_area_stats`: Calculate primary/secondary area splits

## Key Features
- Calculate statistics by category/zone
- Compute continuous data statistics (mean, std, etc.)
- Calculate AGB and carbon statistics with scaling
- Handle area-weighted calculations with flexible units
- Support both per-pixel and constant area values
- Process percentage-based coverage statistics

## Notes
- All values must be in the same CRS
- Negative values in continuous data are treated as no-data
- Category value 0 is treated as no-data in categorical arrays
- AGB calculations use a default scaling factor of 10, must be in "ha"

## Basic Usage

```python
from ctreeskit import XrZonalStats

# Initialize with data arrays
stats = XrZonalStats(
    categorical_da=forest_types,
    continuous_da=biomass_data,
    area_da=pixel_areas
)

# Using constant area value with specified unit
stats = XrZonalStats(
    categorical_da=forest_types,
    continuous_da=biomass_data,
    area_value=100,
    area_unit=Units.HA  # or "ha" as string
)

# Calculate category-specific statistics
category_stats = stats.calculate_categorical_stats()

# Calculate AGB statistics
agb_stats = stats.calculate_agb_stats()

# Calculate statistics by category
detailed_stats = stats.calculate_stats_by_category(agb=True)
```

## Class Methods

### `__init__`

```python
def __init__(
    categorical_da=None,
    continuous_da=None,
    percentage_da=None,
    area_da=None,
    area_value=None,
    area_unit="ha"
)
```

Initialize ZonalStats with input data arrays and area information.

**Parameters:**
- `categorical_da` (xr.DataArray, optional): Categorical mask data
- `continuous_da` (xr.DataArray, optional): Continuous value data
- `percentage_da` (xr.DataArray, optional): Percentage coverage values (0-1)
- `area_da` (xr.DataArray, optional): Area array in hectares -> recommended: `create_weighted_area_geom_mask_da`
- `area_value` (float, optional): Constant area per pixel
- `area_unit` (str | Units): Unit for area_value (default: "ha")
  - Accepted values: "m2", "ha", "km2" or Units enum
  - Only applies to area_value
  - Ignored when using area_da

**Attributes:**
- `categorical_da` (xr.DataArray): Categorical data for zone definitions
  - Values should be integers > 0
  - 0 is treated as no-data
  - None if not provided
- `continuous_da` (xr.DataArray): Continuous data for statistics
  - Used for mean, std, etc. calculations
  - Negative values treated as no-data
  - None if not provided
- `percentage_da` (xr.DataArray): Percentage coverage data
  - Values should be 0-1
  - Used for area weighting
  - None if not provided
- `area_da` (xr.DataArray): Per-pixel areas with units in attrs
  - Must include 'units' attribute ('m²', 'ha', 'km²')
  - Takes precedence over area_value
  - None if not provided
- `area_value` (float): Constant area per pixel
  - Used if area_da is None
  - Units specified by area_unit
  - None if not provided
- `area_unit` (str | Units): Unit for area_value calculations
  - Only applies when using area_value
  - Accepted values: "m2", "ha", "km2" or Units enum
  - Defaults to "ha"
  - Ignored when using area_da

**Raises:**
- ValueError: If input DataArrays have different CRS
- ValueError: If DataArrays don't have CRS information

**Example:**
```python
# Initialize with all optional parameters
stats = XrZonalStats(
    categorical_da=forest_types,
    continuous_da=biomass_data,
    percentage_da=coverage_data,
    area_da=pixel_areas
)

# Initialize with constant area
stats = XrZonalStats(
    categorical_da=forest_types,
    continuous_da=biomass_data,
    area_value=0.5,
    area_unit="ha" # 0.5 hectares per pixel
)
```
### `calculate_categorical_stats()`

Calculate areas or pixel counts for each category.

**Returns:**
- List of dictionaries containing:
  - `category`: category value
  - `pixel_count`: number of pixels
  - `area_{unit}`: area in specified unit (if area provided)
    - Unit from area_da.attrs['units'] or area_unit

### `calculate_continuous_stats(scaling_factor=None)`

Calculate basic statistics for continuous data.

**Parameters:**
- `scaling_factor` (float, optional): Scaling factor to apply to values

**Returns:**
Dictionary containing:
- `count`: Number of valid pixels
- `sum`: Sum of values
- `mean`: Mean value
- `std`: Standard deviation
- `min`: Minimum value
- `max`: Maximum value
- `median`: Median value

### `calculate_agb_stats(scaling_factor=10)`

Calculate AGB and carbon statistics.

**Parameters:**
- `scaling_factor` (int): Scaling factor for AGB values (default: 10)

**Returns:**
Dictionary containing:
- `mean_agb_Mg_per_ha`: Mean AGB in Mg/ha
- `std_agb_Mg_per_ha`: Standard deviation of AGB
- `mean_ton_CO2_per_ha`: Mean CO2 in tons/ha
- `std_ton_CO2_per_ha`: Standard deviation of CO2
- `total_stock_CO2_Mg`: Total CO2 stock (if area provided)
- `area_ha`: Total area in hectares (if area provided)

### `calculate_stats_by_category(scaling_factor=None, agb=True)`

Calculate statistics for each category.

**Parameters:**
- `scaling_factor` (float, optional): Scaling factor for values
- `agb` (bool): Whether to calculate AGB stats (True) or basic stats (False)

**Returns:**
List of dictionaries with statistics per category, including:
- `class`: Category value
- `pixel_count`: Number of pixels
- `area_{unit}`: Area in specified unit (if area provided)
- Statistics from either `calculate_agb_stats()` or `calculate_continuous_stats()`
- Unit from area_da.attrs['units'] or area_unit

### `calculate_percentage_area_stats()`

Calculate primary and secondary area based on percentage values.
**Returns:**
Dictionary containing:
- `primary_area_{unit}`: Area weighted by percentage/100
- `secondary_area_{unit}`: Area weighted by (1 - percentage/100)
- `unit_name`: Name of the area unit used
- `unit_symbol`: Symbol of the area unit used

**Example:**
For a pixel with 75% coverage and area of 1 km²:
```python
{
    'primary_area_km²': 0.75,
    'secondary_area_km²': 0.25,
    'unit_name': 'square kilometers',
    'unit_symbol': 'km²'
}

## License

This project is licensed under the MIT License. See the LICENSE file for details.

# CTrees Tools - Beta Version

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
pip install git+https://github.com/ctrees-products/ctreeskit.git
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
# XrSpatialProcessor

A class for processing spatial data with xarray DataArrays, including area calculations and geometry intersections.

## Key Methods
- `reproject_match_target_da`: Match resolution and extent of target dataset
- `create_binary_geom_mask_da`: Create binary mask (1 inside geometry, 0 outside)
- `create_weighted_geom_mask_da`: Calculate intersection proportions (0-1)
- `create_area_mask_da`: Calculate grid cell areas with flexible units
- `create_weighted_area_geom_mask_da`: Calculate geometry-weighted areas with units (`weighted_geom_mask` * `area_mask`)
- `create_clipped_da_vector`: Clip using vector geometry
- `create_clipped_da_raster`: Clip using raster mask

## Key Features
- Calculate pixel intersection proportions with geometries
- Compute grid cell areas with flexible unit handling (m², ha, km²)
- Handle coordinate transformations and projections
- Mask and weight by geometry intersections

## Notes
- Area calculations support multiple units (m², ha, km²)
- Units are preserved in output DataArrays
- Geometries must be in the same CRS as the base dataset
- Area calculations assume planar coordinates
- Geometry processing handles both single and multiple features

## Usage

```python
from ctreeskit import XrSpatialProcessor

# Initialize with base dataset only
processor = XrSpatialProcessor(base_dataset)

# Initialize with geometry file
processor = XrSpatialProcessor(
    base_dataset,
    geom_source="area.geojson",
    dissolve=True
)

# get attributes
da = processor.da
geom = processor.geom[0]
bbox = processor.geom_bbox

# Calculate masked areas
masked = processor.create_binary_geom_mask_da()
weights = processor.create_weighted_geom_mask_da()
areas = processor.create_weighted_area_geom_mask_da()
```

## Class Methods

### init
### `__init__(base_da, geom_source=None, dissolve=True)`

Initialize SpatialProcessor with a base dataset and optional geometry.

**Parameters:**
- `base_da` (xr.DataArray): Base dataset for processing
- `geom_source` (Union[str, gpd.GeoDataFrame], optional): Input geometry as either:
  - Path to GeoJSON/GPKG file (str)
  - GeoDataFrame (gpd.GeoDataFrame)
- `dissolve` (bool): Whether to dissolve all geometries into one (default: True)

**Attributes:**
- `da` (xr.DataArray): The base dataset for processing
- `geom` (List[shapely.Geometry]): List of geometries for processing
  - Single geometry if dissolve=True
  - Multiple geometries if dissolve=False
  - None if no geometry provided
- `geom_bbox` (tuple): Bounds of geometry (minx, miny, maxx, maxy)
  - Used for efficient clipping operations
  - None if no geometry provided

**Raises:**
- ValueError: If CRS don't match between dataset and geometry
- ValueError: If geometry source has no CRS information
- ValueError: If base dataset has no CRS information
- ValueError: If geometry source is not a file path or GeoDataFrame

**Example:**
```python
# Initialize with file path
processor = XrSpatialProcessor(
    base_dataset,
    geom_source="area.geojson",
    dissolve=True
)

# Access attributes
print(processor.geom_bbox)  # Returns (minx, miny, maxx, maxy)
print(len(processor.geom))  # Returns 1 if dissolved, N if not dissolved
```

### `reproject_match_target_da(target_da)`

Reproject and resample source DataArray to match target DataArray's resolution and extent.

**Parameters:**
- `target_da` (xr.DataArray): Target data whose grid to match

**Returns:**
- xr.DataArray: Reprojected data matching target grid

### `create_binary_geom_mask_da(fit_to_geometry=True)`

Create binary mask from geometry.

**Parameters:**
- `fit_to_geometry` (bool): Match geometry extent (default: True)

**Returns:**
- xr.DataArray: Binary mask (1=inside geometry, 0=outside)

### `create_weighted_geom_mask_da()`

Calculate pixel intersection proportions.

**Returns:**
- xr.DataArray: Intersection proportions (0-1)

### `create_area_mask_da(input_projection=4236, area_unit="ha")`

Calculate grid cell areas with flexible unit specification.

**Parameters:**
- `input_projection` (int): EPSG code (default: 4236)
- `area_unit` (str | Units): Output unit (default: "ha")
  - Accepted values: "m2", "ha", "km2" or Units enum

**Returns:**
- xr.DataArray: Areas in specified unit with unit information in attrs


### `create_weighted_area_geom_mask_da(input_projection=4236, area_unit="ha")`

Calculate geometry-weighted areas with flexible unit specification.

**Parameters:**
- `input_projection` (int): EPSG code (default: 4236)
- `area_unit` (str | Units): Output unit (default: "ha")
  - Accepted values: "m2", "ha", "km2" or Units enum

**Returns:**
- xr.DataArray: Weighted areas in specified unit with unit information in attrs

### `create_clipped_da_vector()`

Clip using vector geometry.

**Returns:**
- xr.DataArray: Clipped dataset

### `create_clipped_da_raster()`

Clip using binary mask.

**Returns:**
- xr.DataArray: Clipped dataset


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

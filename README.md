# CTrees Tools - Beta Version

## Open Source Components (ctreeskit-core)
(( TO BE IMPLEMENTED))
```bash
pip install ctreeskit
```

## Quick Links
- [Installation Guide](#installation)
- [Xr Analyzer ReadMe](./docs/xr_analyzer.md)
- [Arraylake Tools ReadMe](./docs/arraylake_tools.md)

## Table of Contents
1. [Installation](#installation)
   - [PyPI Installation](#from-pypi)
   - [GitHub Installation](#from-github)
   - [Development Setup](#development-installation)
   - [Testing](#testing)
   - [Contributing](#contributing)
2. [Features](#features)
3. [API Reference](#api-reference)
    - [XR Spatial Processor Overview](#xrspatialprocessor)
    - [XR Zonal Stats Overview](#xrzonalstats)
    - [Arraylake Tools Overview](#arraylaketools)

## Installation

You can install ctreeskit either from PyPI or directly from GitHub:

### From PyPI
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
git checkout module_base_package

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install in editable mode with all dependencies
pip install -e ".[all]"
```

## Dependencies

- xarray
- rioxarray (for spatial operations)
- numpy
- shapely
- pyproj
- s3fs (for Amazon S3 storage access)
- python > 3.11

## Testing

To run the tests for the ctreeskit package, navigate to the project directory and execute:

```bash
pytest -m tests
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## Features

- Spatial processing with xarray DataArrays
- Area calculations and geometry data
- Arraylake Ingestion tools 

A Python module for efficient geospatial operations on raster data using xarray, with support for integrating vector geometries and calculating areas.

## API Reference

# XrSpatialProcessor
[Xr Spatial Processor ReadMe](./docs/xr_analyzer.md#xr_spatial_processor_module)

This module provides tools to:

- Process geospatial vector data (from files or objects) into standardized geometry containers
- Clip rasters to geometries or bounding boxes
- Align and resample rasters to match reference grids
- Calculate accurate cell areas for geographic rasters
- Create weighted masks based on geometry-pixel intersections

# XrZonalStats
[Xr Zonal Stats ReadMe](./docs/xr_analyzer.md#xr_zonal_stats_module)

This module provides tools to:

- Calculate area statistics for different classes in categorical rasters
- Calculate area statistics for a combination of two categorical rasters combined
- Support both time-series and static (non-temporal) raster data
- Offer flexible area calculation options (pixel counts, constant values, or spatially-variable areas)
- Generate tabular summaries as pandas DataFrames

# ArraylakeTools
[Arraylake Tools ReadMe](./docs/arraylake_tools.md)

This module provides tools to:

- Calculate area statistics for different classes in categorical rasters
- Support both time-series and static (non-temporal) raster data
- Offer flexible area calculation options (pixel counts, constant values, or spatially-variable areas)
- Generate tabular summaries as pandas DataFrames

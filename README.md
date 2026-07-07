# CTrees Tools - Beta Version
CTreesKit is a public package used split into two sections, "arraylake_tools" which allows a simplified way of converting from geotiffs -> zarr format and saving the data into [arraylake (by Earthmover)](https://docs.earthmover.io/concepts/overview). "xr_analyzer" is a small wrapper for xarray functions used for zonal stats that use the arraylake datasource as the input. These can also be used with Earthmover's opensource [icechunk format](https://icechunk.io/en/latest/overview/) as well! 
Tutorials to come in May 2025 with additional functionality. 
[Slide Deck for CNG Conference about this pip package](https://drive.google.com/file/d/10UO7PcYldF-FdihrmBiYmjsGXC1EHRHm/view?usp=sharing) 

## Open Source Components (ctreeskit-core)
Current:
```bash
pip install git+https://github.com/ctrees-products/ctreeskit.git
```

(( Future -> To Be Implemented June 2025))
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
The project is managed with [uv](https://docs.astral.sh/uv/). `uv sync` creates the
virtual environment and installs the locked dependencies:
```bash
# Clone the repository
git clone https://github.com/ctrees-products/ctreeskit.git
cd ctreeskit

# Core + interactive extra + dev tooling (pip-only, no GDAL)
uv sync --extra interactive --group dev
```

The heavy zonal-stats-on-dask backend (`ctreeskit.dask_analyzer`) lives behind the
optional `zonal` extra. Its pip deps install with `uv sync --extra zonal`, but the
subpackage also needs the GDAL Python bindings (`osgeo`), which must come from
conda-forge and match your system libgdal:
```bash
conda install -c conda-forge gdal   # provides osgeo matching system libgdal
uv sync --extra zonal               # exactextract, odc-geo, distributed
```

## Dependencies

- Python >= 3.12
- xarray / rioxarray / rasterio (spatial operations)
- numpy / pandas / scipy / shapely / pyproj
- s3fs (Amazon S3 access)
- arraylake, icechunk (>=2.0.6), zarr (>=3.1.0) for versioned Icechunk repos
- dask (chunked array processing)

## Testing

The tests run under uv:

```bash
uv run pytest tests/
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

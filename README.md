# CTrees Tools - Beta Version

[![Documentation Status](https://readthedocs.org/projects/ctreeskit/badge/?version=latest)](https://ctreeskit.readthedocs.io)

CTreesKit is a public package used split into two sections, "arraylake_tools" which allows a simplified way of converting from geotiffs -> zarr format and saving the data into [arraylake (by Earthmover)](https://docs.earthmover.io/concepts/overview). "xr_analyzer" is a small wrapper for xarray functions used for zonal stats that use the arraylake datasource as the input. These can also be used with Earthmover's opensource [icechunk format](https://icechunk.io/en/latest/overview/) as well! 
[Slide Deck for CNG Conference about this pip package](https://drive.google.com/file/d/10UO7PcYldF-FdihrmBiYmjsGXC1EHRHm/view?usp=sharing) 

📖 **Full documentation: [ctreeskit.readthedocs.io](https://ctreeskit.readthedocs.io)**

## Open Source Components (ctreeskit-core)
```bash
pip install git+https://github.com/ctrees-products/ctreeskit.git
pip install ctreeskit
```

## Quick Links
- [Installation Guide](#installation)
- [Spatial processing guide](https://ctreeskit.readthedocs.io/en/latest/guides/spatial.html)
- [Zonal statistics guide](https://ctreeskit.readthedocs.io/en/latest/guides/zonal_stats.html)
- [Icechunk ingestion guide](https://ctreeskit.readthedocs.io/en/latest/guides/ingestion.html)
- [API reference](https://ctreeskit.readthedocs.io/en/latest/api/spatial.html)

## Table of Contents
1. [Installation](#installation)
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

# Core + dev tooling (pip-only, no GDAL)
uv sync --group dev
```

Coverage-fraction rasterization (geometry-weighted zonal stats, via
[rasterix](https://rasterix.readthedocs.io/) + exactextract) lives behind the
optional `zonal` extra — fully pip-installable, no GDAL required:
```bash
uv sync --extra zonal               # rasterix, exactextract, geopandas, sparse
```
See the [coverage-fraction example](https://ctreeskit.readthedocs.io/en/latest/guides/zonal_stats.html#coverage-fraction-weighted-statistics)
in the docs. Note: the extra's numba dependency caps numpy below 2.5, so installs
with the `zonal` extra resolve numpy to a 2.4.x release.

Reading dataset configs or GeoJSON geometries from `s3://` paths requires the
optional `s3` extra (boto3, using the standard AWS credential chain):
```bash
uv sync --extra s3
```

## Dependencies

- Python >= 3.12
- xarray / rioxarray / rasterio (spatial operations)
- numpy / pandas / shapely / pyproj
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

Full narrative guides and an auto-generated API reference live at
**[ctreeskit.readthedocs.io](https://ctreeskit.readthedocs.io)**.

### [Spatial processing](https://ctreeskit.readthedocs.io/en/latest/guides/spatial.html) (`xr_analyzer`)

- Process geospatial vector data (from files or objects) into standardized geometry containers
- Clip rasters to geometries or bounding boxes
- Align and resample rasters to match reference grids
- Calculate accurate cell areas for geographic rasters
- Create weighted masks based on geometry-pixel intersections

### [Zonal statistics](https://ctreeskit.readthedocs.io/en/latest/guides/zonal_stats.html) (`xr_analyzer`)

- Calculate area statistics for different classes in categorical rasters
- Calculate area statistics for a combination of two categorical rasters combined
- Support both time-series and static (non-temporal) raster data
- Offer flexible area calculation options (pixel counts, constant values, or spatially-variable areas)
- Generate tabular summaries as pandas DataFrames

### [Icechunk ingestion](https://ctreeskit.readthedocs.io/en/latest/guides/ingestion.html) (`arraylake_tools`)

- Add CF-compliant metadata to a dataset from a dataset configuration
- Allocate a lazy `(time, y, x)` schema template, then populate it with annual raster data
- Ingest annual GeoTIFF/VRT mosaics with Dask-backed, chunked writes
- Region-write or append each year onto a versioned Icechunk time axis

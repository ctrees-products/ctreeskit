# ctreeskit

`ctreeskit` is a lightweight, pip-installable geospatial toolkit for
[xarray](https://docs.xarray.dev/)-based raster analysis and for ingesting raster
mosaics into versioned [Icechunk](https://icechunk.io/) data repositories.

It is organised into two subpackages:

- **`xr_analyzer`** — geometry handling, raster clipping and reprojection, accurate
  grid-cell areas, and categorical zonal statistics returned as tidy pandas
  DataFrames.
- **`arraylake_tools`** — a dataset-configuration handler that adds CF-compliant
  metadata to a Dataset, and an annual-mosaic ingestion connector that writes
  GeoTIFF/VRT mosaics into a versioned `(time, y, x)` Icechunk layout.

The core install pulls in no GDAL and no cloud SDK — those are opt-in extras — so it
stays easy to add to any environment.

## Quick start

```bash
pip install "ctreeskit @ git+https://github.com/ctrees-products/ctreeskit.git"
```

```python
import xarray as xr
from ctreeskit import (
    process_geometry,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    calculate_categorical_area_stats,
)

# A categorical (land-cover-style) raster in WGS 84.
ds = xr.open_dataarray("landcover.tif")

# Clip it to an area of interest, then tabulate area per class.
aoi = process_geometry("aoi.geojson")
clipped = clip_ds_to_geom(ds, aoi)
area = create_area_ds_from_degrees_ds(clipped)

stats = calculate_categorical_area_stats(clipped, area_ds=area)
stats.to_csv("area_by_class.csv")
```

See {doc}`installation` for the optional extras, and the guides below for the full
workflow.

```{toctree}
:maxdepth: 2
:hidden:
:caption: User guide

installation
guides/spatial
guides/zonal_stats
guides/ingestion
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: API reference

api/spatial
api/zonal
api/arraylake
```

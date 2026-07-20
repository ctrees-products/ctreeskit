# Spatial processing

The `xr_analyzer` spatial tools turn vector geometries and rasters into aligned,
analysis-ready inputs: loading and dissolving geometries, clipping rasters,
matching grids, and computing accurate per-cell areas. Every function is available
from the top-level package.

## Loading geometries

{func}`~ctreeskit.process_geometry` reads a geometry source — a GeoJSON path (local
or `s3://`), a single Shapely geometry, or a list of geometries — and returns a
{class}`~ctreeskit.GeometryData` container holding the (optionally dissolved)
geometries, their CRS, bounding box, and computed area.

```python
from ctreeskit import process_geometry

aoi = process_geometry("aoi.geojson", dissolve=True, output_in_ha=True)
aoi.geom_bbox   # (minx, miny, maxx, maxy)
aoi.geom_area   # area in hectares
```

GeoJSON coordinates are WGS 84 by definition (RFC 7946); a legacy `crs` member is
honoured when present, otherwise `EPSG:4326` is assumed. Reading from an `s3://`
path requires the [`s3` extra](s3-extra).

## Clipping rasters

Two clippers cover the common cases:

- {func}`~ctreeskit.clip_ds_to_bbox` clips to a bounding box `(minx, miny, maxx, maxy)`.
  Pass `drop_time=True` to keep only the first time slice of a time series.
- {func}`~ctreeskit.clip_ds_to_geom` clips to the geometry itself. It accepts either a
  {class}`~ctreeskit.GeometryData` or any raw geometry source (which it processes for
  you). Set `all_touch=True` to include every pixel the geometry boundary touches.

```python
from ctreeskit import clip_ds_to_bbox, clip_ds_to_geom

by_bbox = clip_ds_to_bbox(ds, aoi.geom_bbox)
by_geom = clip_ds_to_geom(ds, aoi)
```

## Matching grids

{func}`~ctreeskit.reproject_match_ds` aligns and resamples a target raster onto a
template raster's grid. By default it also returns a matching grid of cell areas:

```python
from ctreeskit import reproject_match_ds

aligned, area_grid = reproject_match_ds(template_raster, target_raster)
```

## Cell areas

For rasters on a geographic (degrees) grid, {func}`~ctreeskit.create_area_ds_from_degrees_ds`
returns a DataArray of per-cell areas. Cell area varies with latitude, so this is
the correct weight for any area-based summary. By default it selects its method from
the data's latitude range — a geodesic calculation poleward of 70°, and an
equal-area projection (EPSG:6933) elsewhere; set `high_accuracy=True` or `False` to
force one.

```python
from ctreeskit import create_area_ds_from_degrees_ds

area = create_area_ds_from_degrees_ds(ds, output_in_ha=True)
```

## Coverage-fraction masks

{func}`~ctreeskit.create_proportion_geom_mask` returns a mask where each pixel value
is the fraction (0–1) of that pixel covered by the geometry, so boundary pixels
contribute proportionally rather than all-or-nothing. When pixels are very small
relative to the geometry, edge effects are negligible and a binary mask is returned
instead; pass `overwrite=True` to always compute fractions.

```python
from ctreeskit import create_proportion_geom_mask

mask = create_proportion_geom_mask(ds, aoi, overwrite=True)
```

For the same idea applied directly to zonal area statistics — and a faster,
dask-aware, sparse implementation — see
{doc}`zonal_stats` and its coverage-fraction section.

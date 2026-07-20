# Zonal statistics

The `xr_analyzer` zonal tools tabulate area by class for categorical rasters,
with or without a time dimension, returning tidy pandas DataFrames. Aggregation
runs as a single [flox](https://flox.readthedocs.io/)-accelerated, dask-compatible
groupby, so chunked rasters stay lazy.

## Area per class

{func}`~ctreeskit.calculate_categorical_area_stats` sums pixel area for each class in
a categorical raster. The `area_ds` argument controls the per-pixel weight:

- an {class}`xarray.DataArray` of per-cell areas (e.g. from
  {func}`~ctreeskit.create_area_ds_from_degrees_ds`),
- a single number applied to every pixel,
- `True` to derive areas from the raster's degree coordinates automatically,
- or omitted to count pixels.

```python
from ctreeskit import calculate_categorical_area_stats

# Weight by true per-cell area derived from the grid.
stats = calculate_categorical_area_stats(ds, area_ds=True)
```

Class `0` is treated as no-data and excluded by default (`drop_zero=True`). When the
raster carries CF `flag_values`/`flag_meanings` metadata, those meanings become the
class labels in the output. By default the table is pivoted wide (one column per
class); pass `reshape=False` for long format.

## Combining two classifications

{func}`~ctreeskit.calculate_combined_categorical_area_stats` cross-tabulates area over
the unique combinations of two categorical rasters — for example land-cover class
against a change flag. The inputs are aligned spatially for you.

```python
from ctreeskit import calculate_combined_categorical_area_stats

stats = calculate_combined_categorical_area_stats(primary_ds, secondary_ds, area_ds=True)
```

Internally, {func}`~ctreeskit.create_combined_classification` encodes each class pair
as a single integer (`primary * modulus + secondary`, with the modulus stored in the
`combined_modulus` attribute); you can call it directly if you want the combined
raster rather than the summary table.

## Coverage-fraction weighted statistics

For area statistics weighted by *how much* of each pixel a geometry covers — rather
than a binary in/out mask — use [rasterix](https://rasterix.readthedocs.io/)'s
exactextract-backed coverage grids, installed with the
[`zonal` extra](zonal-extra).
The coverage grid is stored sparse (only touched pixels are materialised) and the
computation is dask-aware, so chunked rasters stay lazy and no full-size dense mask
is written.

```python
import geopandas as gpd
from rasterix.rasterize.exact import coverage

from ctreeskit import calculate_categorical_area_stats, create_area_ds_from_degrees_ds

classes = ...  # categorical (y, x) DataArray with CRS metadata
geoms = gpd.read_file("aoi.geojson")

# Fraction of each pixel covered by the geometry: dims (geometry, y, x).
cover = coverage(classes, geoms, coverage_weight="fraction")

# Densify one geometry's coverage grid and weight per-pixel areas by it.
cover_2d = cover.isel(geometry=0)
cover_2d = cover_2d.copy(data=cover_2d.data.todense())
area = create_area_ds_from_degrees_ds(classes)

stats = calculate_categorical_area_stats(classes, area_ds=area * cover_2d)
```

Edge pixels contribute only their covered fraction, so `total_area` matches the
geometry's true footprint instead of over-counting boundary pixels.

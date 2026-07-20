# Icechunk ingestion

The `arraylake_tools` subpackage writes annual raster mosaics into a versioned
[Icechunk](https://icechunk.io/) repository with a `(time, y, x)` layout, and adds
CF-compliant metadata to datasets along the way.

## Dataset configuration

{class}`~ctreeskit.ArraylakeDatasetConfig` loads a dataset configuration — either from
a config store (a bucket of JSON configs, which requires the
[`s3` extra](s3-extra)) or from a dict you pass
directly — and applies standardized CF metadata to an xarray Dataset: coordinate
attributes, `flag_values`/`flag_meanings` for classification variables, and
grid-mapping links.

```python
from ctreeskit import ArraylakeDatasetConfig

config = ArraylakeDatasetConfig(dataset_name="example_dataset_name")
ds = config.add_cf_metadata(ds)
```

Configuration dictionaries follow a JSON template; bundled example templates ship
with the package and can be loaded with `load_config`.

## Annual raster ingestion

{class}`~ctreeskit.AnnualRasterIngester` ingests annual GeoTIFF/VRT mosaics into an
Icechunk repository. It separates schema allocation from data population:

- `initialize_schema` allocates the full time axis once as an empty, lazy template —
  coordinates and metadata only, with no data chunks written.
- `ingest_year` then fills one year at a time, either as a region write into its
  pre-allocated slot or as an append when the year extends the time axis. Only
  non-nodata chunks are written to storage.

```python
from ctreeskit.arraylake_tools.ingest import load_config
from ctreeskit import AnnualRasterIngester

config = load_config("example_config_name")  # or your own config dict
ingester = AnnualRasterIngester(config, token="your_api_token")

ingester.create_repo()
ingester.initialize_schema()
ingester.ingest_year(2020)
ingester.ingest_year(2021, tag="2021-release")
```

### Targeting a branch

Pass `branch_name` (default `"main"`) to point every operation — schema
initialization and per-year ingestion — at a specific Icechunk branch, such as a
disposable test branch instead of production:

```python
ingester = AnnualRasterIngester(
    config, token="your_api_token", branch_name="test-ingest"
)
```

### Ingesting a spatial subset

`ingest_year` accepts a `bbox=(minx, miny, maxx, maxy)` window (in the dataset CRS)
to ingest a spatial subset for fast verification, without writing the full extent.
The window must align with the stored grid — same resolution and origin — and an
off-grid bbox raises rather than writing shifted pixels.

```python
ingester.ingest_year(2020, bbox=(-60.0, -10.0, -55.0, -5.0))
```

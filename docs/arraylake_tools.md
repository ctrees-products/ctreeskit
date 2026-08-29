# Arraylake Tools

The `arraylake_tools` package provides utilities for ingesting raster data into Arraylake/Icechunk repositories: a dataset-configuration handler (`ArraylakeDatasetConfig`) and an annual-mosaic ingestion connector (`AnnualRasterIngester`) that writes GeoTIFF/VRT mosaics into a versioned `(time, y, x)` layout.

## Table of Contents

- [Usage](#usage)
    - [Dataset Configuration](#dataset-configuration)
    - [Annual Raster Ingestion](#annual-raster-ingestion)
- [Modules](#modules)
    - [common.py](#commonpy)
    - [ingest.py](#ingestpy)

## Usage

### Dataset Configuration

The `ArraylakeDatasetConfig` class loads a dataset configuration (from an S3 bucket of configs — requires the optional `s3` extra — or a dict passed directly) and adds CF-compliant metadata — coordinate attributes, `flag_values`/`flag_meanings` for classification variables, grid-mapping links — to an xarray Dataset.

```python
from ctreeskit import ArraylakeDatasetConfig

config = ArraylakeDatasetConfig(dataset_name="your_dataset_name")
ds = config.add_cf_metadata(ds)
```

### Annual Raster Ingestion

The `AnnualRasterIngester` class ingests annual GeoTIFF/VRT mosaics into an
Arraylake/Icechunk repo with a `(time, y, x)` layout. It separates schema
allocation from data population: `initialize_schema` allocates the full time axis
once as an empty, lazy template (no data chunks written); `ingest_year` then fills
one year at a time, either as a region write into its pre-allocated slot or an
append when the year extends the time axis.

```python
from ctreeskit.arraylake_tools.ingest import AnnualRasterIngester, load_config

config = load_config("your_config_name")  # or your own config dict
ingester = AnnualRasterIngester(config, token="your_arraylake_api_token")

ingester.create_repo()
ingester.initialize_schema()
ingester.ingest_year(2020)
ingester.ingest_year(2021, tag="2021-release")
```

Pass `branch_name` (default `"main"`) to point every operation -- schema
initialization and per-year ingestion -- at a specific Icechunk branch, e.g. a
disposable test branch instead of production:

```python
ingester = AnnualRasterIngester(config, token="your_arraylake_api_token", branch_name="test-ingest")
```

`ingest_year` also accepts a `bbox=(minx, miny, maxx, maxy)` window (in the dataset
CRS) to ingest a spatial subset for fast verification, without ingesting the full
extent. The window must align with the stored grid (same resolution and origin);
an off-grid bbox raises rather than writing shifted pixels.

## Modules
### common.py

This module contains the `ArraylakeDatasetConfig` class, which handles dataset configuration loading and validation from a config file. It provides helper properties and methods to extract and add standardized metadata to an xarray Dataset based on configuration information.

### ingest.py

This module contains the `AnnualRasterIngester` class and `load_config` helper. `AnnualRasterIngester` reads finished per-year GeoTIFF/VRT mosaics from S3 and writes them into an Arraylake/Icechunk repo with a `(time, y, x)` layout: `initialize_schema` allocates the full time axis once as an empty, lazy template (coordinates + metadata only, no data chunks); `ingest_year` fills one year at a time via a region write into its pre-allocated slot, or an `append_dim="time"` append for a year beyond the current axis. `load_config` loads an example dataset-config template bundled with this package.

# Arraylake Tools

The `arraylake_tools` package provides a set of utilities for interacting with Arraylake repositories. It includes functionality for creating repositories from configurations, initializing repositories (setting schema), populating repositories with raster data using Dask for asynchronous processing, and ingesting annual raster mosaics into a versioned `(time, y, x)` layout.

## Table of Contents

- [Usage](#usage)
    - [Creating Repositories](#creating-repositories)
    - [Initialization](#initialization)
    - [Populating Repositories](#populating-repositories)
    - [Annual Raster Ingestion](#annual-raster-ingestion)
- [Modules](#modules)
    - [common.py](#commonpy)
    - [create.py](#createpy)
    - [initialize.py](#initializepy)
    - [populate_dask.py](#populate_daskpy)
    - [ingest.py](#ingestpy)

## Usage
### Creating Repositories

The `ArraylakeRepoCreator` class simplifies the creation and initialization of Arraylake repositories. It supports direct creation using explicit parameters or automated creation by processing JSON configuration files stored in S3.

```python
from arraylake_tools.create import ArraylakeRepoCreator

creator = ArraylakeRepoCreator(token="your_api_token")
creator.create(dataset_name="your_dataset_name", organization_name="your_organization")
creator.create_from_s3(uri="s3://path/to/config.json")
```

### Initialization

The `ArraylakeRepoInitializer` class is used to initialize an Arraylake repository from a configuration. It can load configuration information from S3 or directly from a dictionary and set up the repository accordingly.

```python
from arraylake_tools.initialize import ArraylakeRepoInitializer

initializer = ArraylakeRepoInitializer(
        token="your_api_token",
        dataset_name="your_dataset_name")
initializer.initialize_all_groups()
```

### Populating Repositories

The `ArraylakeRepoPopulator` class populates groups of an Arraylake repository with raster data. It supports concurrent processing of time-enabled groups and merges the resulting sessions.

```python
from arraylake_tools.populate_dask import ArraylakeRepoPopulator

populator = ArraylakeRepoPopulator(token="your_api_token", dataset_name="your_dataset_name")
populator.populate_all_groups()
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
extent.

## Modules
### common.py

This module contains the `ArraylakeDatasetConfig` class, which handles dataset configuration loading and validation from a config file. It provides helper properties and methods to extract and add standardized metadata to an xarray Dataset based on configuration information.


### create.py

This module contains the `ArraylakeRepoCreator` class, which simplifies the creation and initialization of Arraylake repositories. It supports direct creation using explicit parameters or automated creation by processing JSON configuration files stored in S3.


### initialize.py

This module contains the `ArraylakeRepoInitializer` class, which initializes an Arraylake repository from a configuration. It handles spatial subsetting using geometry from a GeoJSON file and creates an xarray Dataset schema for each group defined in the configuration.

### populate_dask.py

This module provides functionality to process and populate annual raster datasets into an Arraylake repository. It leverages Dask for asynchronous processing and icechunk for writing data in a distributed manner. The `ArraylakeRepoPopulator` class loads a dataset configuration, initializes an Arraylake repository session, and populates each group concurrently.

### ingest.py

This module contains the `AnnualRasterIngester` class and `load_config` helper. `AnnualRasterIngester` reads finished per-year GeoTIFF/VRT mosaics from S3 and writes them into an Arraylake/Icechunk repo with a `(time, y, x)` layout: `initialize_schema` allocates the full time axis once as an empty, lazy template (coordinates + metadata only, no data chunks); `ingest_year` fills one year at a time via a region write into its pre-allocated slot, or an `append_dim="time"` append for a year beyond the current axis. `load_config` loads an example dataset-config template bundled with this package.

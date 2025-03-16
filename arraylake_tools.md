# Arraylake Tools

The `arraylake_tools` package provides a set of utilities for interacting with Arraylake repositories. It includes functionality for initializing repositories, creating repositories from configurations, and populating repositories with raster data using Dask for asynchronous processing.

## Table of Contents

- [Usage](#usage)
    - [Initialization](#initialization)
    - [Creating Repositories](#creating-repositories)
    - [Populating Repositories](#populating-repositories)
- [Modules](#modules)
    - [initialize.py](#initializepy)
    - [common.py](#commonpy)
    - [create.py](#createpy)
    - [populate_dask.py](#populate_daskpy)

## Usage

### Initialization

The `ArraylakeRepoInitializer` class is used to initialize an Arraylake repository from a configuration. It can load configuration information from S3 or directly from a dictionary and set up the repository accordingly.

```python
from arraylake_tools.initialize import ArraylakeRepoInitializer

initializer = ArraylakeRepoInitializer(
        token="your_api_token",
        dataset_name="your_dataset_name")
initializer.initialize_all_groups()
```

### Creating Repositories

The `ArraylakeRepoCreator` class simplifies the creation and initialization of Arraylake repositories. It supports direct creation using explicit parameters or automated creation by processing JSON configuration files stored in S3.

```python
from arraylake_tools.create import ArraylakeRepoCreator

creator = ArraylakeRepoCreator(token="your_api_token")
creator.create(dataset_name="your_dataset_name", organization_name="your_organization")
creator.create_from_s3(uri="s3://path/to/config.json")
```

### Populating Repositories

The `ArraylakeRepoPopulator` class populates groups of an Arraylake repository with raster data. It supports concurrent processing of time-enabled groups and merges the resulting sessions.

```python
from arraylake_tools.populate_dask import ArraylakeRepoPopulator

populator = ArraylakeRepoPopulator(token="your_api_token", dataset_name="your_dataset_name")
populator.populate_all_groups()
```

## Modules

### initialize.py

This module contains the `ArraylakeRepoInitializer` class, which initializes an Arraylake repository from a configuration. It handles spatial subsetting using geometry from a GeoJSON file and creates an xarray Dataset schema for each group defined in the configuration.

### common.py

This module contains the `ArraylakeDatasetConfig` class, which handles dataset configuration loading and validation from a config file. It provides helper properties and methods to extract and add standardized metadata to an xarray Dataset based on configuration information.

### create.py

This module contains the `ArraylakeRepoCreator` class, which simplifies the creation and initialization of Arraylake repositories. It supports direct creation using explicit parameters or automated creation by processing JSON configuration files stored in S3.

### populate_dask.py

This module provides functionality to process and populate annual raster datasets into an Arraylake repository. It leverages Dask for asynchronous processing and icechunk for writing data in a distributed manner. The `ArraylakeRepoPopulator` class loads a dataset configuration, initializes an Arraylake repository session, and populates each group concurrently.

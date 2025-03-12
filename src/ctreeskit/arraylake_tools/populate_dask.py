"""
populate_dask.py

This module provides functionality to process and populate annual raster datasets into an Arraylake repository.
It leverages Dask for asynchronous processing and icechunk for writing data in a distributed manner.

The key functions and classes are:

- process_annual_dataset_fn:
    Processes an annual raster file:
      - Reinitializes the API client and session in the worker.
      - Opens the raster file, casts it to the appropriate data type.
      - Expands dimensions with the appropriate timestamp if a time dimension exists.
      - Cleans up unused variables and problematic attributes.
      - Writes the data to the repository using icechunk.
      - Returns a writable session that can later be merged.

- ArraylakeRepoPopulator:
    A class that loads a dataset configuration (from an S3 dataset name or a given configuration dictionary),
    initializes an Arraylake repository session, and populates each group (and its variables) concurrently.
    The class supports processing of time-enabled groups (annual files) and merges the resulting sessions.
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from icechunk.xarray import to_icechunk
from icechunk.distributed import merge_sessions
from arraylake import Client as arraylakeClient
import rioxarray as rio
import numpy as np
from .common import ArraylakeDatasetConfig


def process_annual_dataset_fn(token: str, repo_name: str, has_time: bool, unit_type: type,
                              year: int, var_name: str, group_name: str, file_uri: str):
    """
    Process one annual raster file and write it to the Arraylake repository.

    Steps performed:
      - Reinitializes the Arraylake API client and retrieves a writable session.
      - Opens a raster file from a given URI using rioxarray.
      - Casts the dataset to the specified data type.
      - If a time dimension is enabled, expands the dataset to include an annual timestamp, 
        using the date "year-01-01".
      - Drops the 'spatial_ref' variable and removes problematic attributes (e.g., add_offset, scale_factor).
      - Writes the processed dataset to the repository via icechunk.
      - Returns the new writable session for later merging.

    Parameters
    ----------
    token : str
        API token used to authenticate with the Arraylake service.
    repo_name : str
        Name of the repository to which the dataset will be written.
    has_time : bool
        Flag indicating if the dataset has a time dimension.
    unit_type : type
        Numpy data type (either np.float32 or np.int16) for casting the raster data.
    year : int
        The year (from the annual file) used for timestamping if time dimension is enabled.
    var_name : str
        The variable name to assign to the dataset.
    group_name : str
        The group name in which this variable will be stored.
    file_uri : str
        S3 URI or local path to the raster file.

    Returns
    -------
    new_session : Session object
        A writable session from the Arraylake repository that contains the written data.
    """
    # Reinitialize client and repo in the worker
    client = arraylakeClient(token=token)
    repo = client.get_repo(repo_name)
    new_session = repo.writable_session("main")

    # Open the file and convert to xarray Dataset using rioxarray.
    ds = rio.open_rasterio(
        file_uri,
        chunks=(1, 4000, 4000),
        lock=False,
        fill_value=-1,  # Set fill value during read
        masked=True     # Ensure proper handling of NoData values
    ).astype(unit_type).to_dataset(name=var_name)
    ds = ds.squeeze("band", drop=True)

    # Define region selection for icechunk processing.
    region = {"x": slice(None), "y": slice(None)}
    if has_time:
        ds = ds.expand_dims(time=[f"{year}-01-01"])
        region = {"time": "auto", "x": slice(None), "y": slice(None)}

    # Remove unused variable and problematic attributes.
    ds = ds.drop_vars(['spatial_ref'])
    for attr in ["add_offset", "scale_factor"]:
        if attr in ds[var_name].attrs:
            del ds[var_name].attrs[attr]

    # Enable session pickling for later merging.
    new_session.allow_pickling()
    to_icechunk(ds.drop_encoding(), new_session,
                group=group_name, region=region)
    return new_session


class ArraylakeRepoPopulator:
    """
    Class for populating groups of an Arraylake repository with raster data.

    This class loads the configuration for a dataset (either by S3 dataset name or by a provided 
    configuration dictionary), initializes an Arraylake repository session, and concurrently processes
    annual (or non-annual) raster files for each variable in a predefined configuration group.
    After processing, individual sessions are merged and the complete changes are committed.

    Attributes
    ----------
    config : dict
        The dataset configuration dictionary.
    dataset_name : str
        The name of the dataset, as defined in the configuration.
    organization : str
        Organization name extracted from the configuration.
    repo_name : str
        The repository name (a combination of organization and dataset_name) where data is stored.
    crs : str
        The coordinate reference system defined in the configuration (default 'EPSG:4326').
    token : str
        API token used for Arraylake repository interactions.
    client : arraylakeClient
        An instance of the Arraylake API client.
    repo : Repository object
        Repository object retrieved from the Arraylake client.
    session : Session object
        A writable session open on the repository.
    groups : dict
        The configuration information for groups (the logical grouping of dataset variables).
    dims : list
        Dataset dimensions, e.g., ['x', 'y'] (and optionally 'time').
    has_time : bool
        Flag indicating whether the dataset includes a time dimension.
    """

    def __init__(self, token: str, dataset_name: str = None, config_dict: dict = None):
        """
        Initialize Populator with a dataset configuration.

        The initializer requires either a dataset_name (to load configuration from S3) or a provided
        configuration dictionary. It sets up client connectivity and defines key properties that dictate 
        how the repository will be populated.

        Parameters
        ----------
        token : str
            Arraylake API token for repository authentication.
        dataset_name : str, optional
            Dataset name used to load the configuration from S3. Mutually exclusive with config_dict.
        config_dict : dict, optional
            A configuration dictionary provided directly. Mutually exclusive with dataset_name.

        Raises
        ------
        ValueError
            If neither dataset_name nor config_dict is provided, or if both are provided.
        """
        if dataset_name is None and config_dict is None:
            raise ValueError("Must provide either dataset_name or config_dict")
        if dataset_name is not None and config_dict is not None:
            raise ValueError(
                "Cannot provide both dataset_name and config_dict")

        # Load configuration from S3 using ArraylakeDatasetConfig if dataset_name is provided.
        if dataset_name is not None:
            config_loader = ArraylakeDatasetConfig(dataset_name)
            self.config = config_loader._config
        else:
            self.config = config_dict

        self.dataset_name = self.config.get('dataset_name')
        self.organization = self.config.get("organization")
        self.repo_name = self.config.get(
            "repo", f"{self.organization}/{self.dataset_name}")
        self.crs = self.config.get('crs', 'EPSG:4326')
        self.token = token

        # Initialize Arraylake client and repository session.
        self.client = arraylakeClient(token=token)
        self.repo = self.client.get_repo(self.repo_name)
        self.session = self.repo.writable_session("main")
        self.groups = self.config.get('groups', {})

        # Setup dimensions.
        self.dims = self.config.get('dim', ['x', 'y'])
        self.has_time = 'time' in self.dims

    def populate_group(self, group_name: str) -> None:
        """
        Populate the specified configuration group with raster data.

        For each variable (except 'time') defined in the group's configuration, this method determines
        the appropriate file URI (constructed using S3 path prefixes/suffixes or a fixed S3 path), then submits
        processing tasks concurrently via a ThreadPoolExecutor. The function waits for all tasks to finish,
        merges the resulting repository sessions, and finally commits the changes.

        Parameters
        ----------
        group_name : str
            The name of the group to populate.

        Raises
        ------
        ValueError
            If the specified group is not found in the configuration.
        """
        if group_name not in self.config['groups']:
            raise ValueError(f"Group {group_name} not found in config.")
        group_config = self.config['groups'][group_name]
        time_config = group_config.get("time", None)
        years = []
        if time_config:
            start_date = pd.Timestamp(time_config.get("start", "2000-01-01"))
            end_date = pd.Timestamp(time_config.get("end", "2024-01-01"))
            freq = time_config.get("freq", "YS")
            years = pd.date_range(start_date, end_date,
                                  freq=freq).year.tolist()
        futures = []
        with ThreadPoolExecutor() as executor:
            # Loop through each variable: if time is enabled, process for each year.
            for var_name, var_config in group_config.items():
                if time_config:
                    s3_path_prefix = var_config.get("s3_path_prefix")
                    s3_path_suffix = var_config.get("s3_path_suffix")
                    for year in years:
                        file_uri = f"{s3_path_prefix}{year}{s3_path_suffix}"
                        print(
                            f"Dispatching task for group '{group_name}', variable '{var_name}', year {year}: {file_uri}")
                        future = executor.submit(
                            process_annual_dataset_fn,
                            self.token,
                            self.repo_name,
                            self.has_time,
                            np.float32 if var_config['unit_type'] == 'float' else np.int16,
                            year,
                            var_name,
                            group_name,
                            file_uri,
                        )
                        futures.append(future)
                else:
                    # If time is not enabled, use a single S3 path.
                    s3_path = var_config.get("s3_path")
                    future = executor.submit(
                        process_annual_dataset_fn,
                        self.token,
                        self.repo_name,
                        self.has_time,
                        np.float32 if var_config['unit_type'] == 'float' else np.int16,
                        0,  # Year is irrelevant without time dimension.
                        var_name,
                        group_name,
                        s3_path,
                    )
                    futures.append(future)
            # Wait for all tasks to complete.
            results = [future.result() for future in futures]

        # Merge all writable sessions returned by the processing tasks.
        self.session = merge_sessions(self.session, *results)
        print(f"Populated group {group_name} with task results: {results}")
        self.session.commit(
            f"Populated group {group_name} with task results: {results}")

    def populate_all_groups(self) -> None:
        """
        Iterate over every group specified in the configuration and populate each.

        This method loops through the groups defined in the configuration and calls populate_group() for each,
        effectively processing all variables for the entire dataset.
        """
        for group in self.config.get("groups", {}).keys():
            print(f"Populating group: {group}")
            self.populate_group(group)

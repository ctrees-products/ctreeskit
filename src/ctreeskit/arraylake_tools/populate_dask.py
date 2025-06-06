from ctreeskit import ArraylakeDatasetConfig
from math import ceil

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
from dask.distributed import LocalCluster, Client
from dask import delayed, compute
from concurrent.futures import ThreadPoolExecutor
from icechunk.xarray import to_icechunk
from icechunk.distributed import merge_sessions
from arraylake import Client as arraylakeClient
import rioxarray as rio
import numpy as np
import dask
import logging
import psutil
import dotenv
dotenv.load_dotenv()


def process_chunk(session, has_time: bool, unit_type: type,
                  var_name: str, group_name: str, ds, chunk: dict):
    """
    Process a chunk of the raster file and write it to the Arraylake repository.

    Parameters
    ----------
    session : Session object
        A writable session passed from populate_group.
    has_time : bool
        Flag indicating if the dataset has a time dimension.
    unit_type : type
        Numpy data type (either np.float32 or np.int16) for casting the raster data.
    var_name : str
        The variable name to assign to the dataset.
    group_name : str
        The group name in which this variable will be stored.
    ds : xarray.Dataset
        The opened dataset passed from populate_group.
    chunk : dict
        Dictionary specifying the chunk boundaries (e.g., {"x": slice(start_x, end_x), "y": slice(start_y, end_y)}).

    Returns
    -------
    None
    """
    # Select the chunk region
    ds_chunk = ds.isel(**chunk)

    # Define region selection for icechunk processing.
    # region = {"x": slice(None), "y": slice(None)}
    # if has_time:
    #     region["time"] = "auto"

    # Remove unused variable and problematic attributes.
    if 'spatial_ref' in ds_chunk:
        ds_chunk = ds_chunk.drop_vars(['spatial_ref'])
    for attr in ["add_offset", "scale_factor"]:
        if attr in ds_chunk[var_name].attrs:
            del ds_chunk[var_name].attrs[attr]

    # Write data to the session.
    to_icechunk(ds_chunk.drop_encoding(), session,
                group=group_name, region="auto")


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

    def __init__(self, token: str, dataset_name: str = None, config_dict: dict = None, max_workers: int = 4):
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
        max_workers : int, optional
            Number of worker threads for concurrent processing. Default is 4.

        Raises
        ------
        ValueError
            If neither dataset_name nor config_dict is provided, or if both are provided.
        """
        dask.config.set(
            {
                "distributed.worker.memory.target": float(
                    os.getenv("DASK_WORKER_MEMORY_TARGET", 0.6)
                ),
                "distributed.worker.memory.spill": float(
                    os.getenv("DASK_WORKER_MEMORY_SPILL", 0.7)
                ),
                "distributed.worker.memory.pause": float(
                    os.getenv("DASK_WORKER_MEMORY_PAUSE", 0.8)
                ),
                "distributed.worker.memory.terminate": float(
                    os.getenv("DASK_WORKER_MEMORY_TERMINATE", 0.95)
                ),
                "distributed.comm.compression": os.getenv("DASK_COMPRESSION", "auto"),
                "distributed.scheduler.work-stealing": os.getenv(
                    "DASK_WORK_STEALING", "true"
                ),
                "distributed.worker.memory.rebalance.measure": os.getenv(
                    "DASK_REBALANCE_MEASURE", "managed_in_memory"
                ),
                "distributed.worker.memory.spill-to-disk": os.getenv(
                    "DASK_SPILL_TO_DISK", "true"
                ),
                "distributed.worker.memory.target-fraction": float(
                    os.getenv("DASK_WORKER_MEMORY_TARGET_FRACTION", 0.8)
                ),
                "distributed.worker.memory.monitor-interval": os.getenv(
                    "DASK_WORKER_MEMORY_MONITOR_INTERVAL", "100ms"
                ),
                "array.slicing.split_large_chunks": os.getenv(
                    "DASK_SPLIT_LARGE_CHUNKS", "true"
                ),
            }
        )
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
        self.max_workers = max_workers

        # Initialize Arraylake client and repository session.
        self.client = arraylakeClient(token=token)
        self.repo = self.client.get_repo(self.repo_name)
        self.session = self.repo.writable_session("main")
        self.groups = self.config.get('groups', {})

        # Setup dimensions.
        self.dims = self.config.get('dim', ['x', 'y'])
        self.has_time = 'time' in self.dims

        # Dynamically configure Dask cluster based on system resources
        total_ram = psutil.virtual_memory().total  # Total RAM in bytes
        cpu_cores = psutil.cpu_count(logical=True)  # Total logical CPU cores

        workers = cpu_cores  # Use full CPU cores as workers
        # 90% of total RAM divided by workers
        memory_limit = round(0.9 * total_ram / workers)
        # Convert to human-readable format
        memory_limit_str = f"{memory_limit // (1024 ** 3)}GB"

        # Initialize Dask LocalCluster and Client
        self.cluster = LocalCluster(
            n_workers=int(os.getenv("DASK_N_WORKERS")),
            threads_per_worker=int(os.getenv("DASK_THREADS_PER_WORKER")),
            memory_limit=os.getenv("DASK_MEMORY_LIMIT"),
            silence_logs=logging.WARNING,
            dashboard_address=f":{os.getenv('DASK_DASHBOARD_PORT')}",
            protocol="tcp://",
        )
        self.dask_client = Client(self.cluster)
        print(f"Dask dashboard available at: {self.cluster.dashboard_link}")
        print(f"Workers: {workers}, Memory per worker: {memory_limit_str}")

    def populate_group(self, group_name: str, unit_type: str = None) -> None:
        """
        Populate the specified configuration group with raster data, submitting tasks for each chunk.
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
        tasks = []

        for var_name, var_config in group_config.items():
            if var_name == "time":
                continue
            if unit_type:
                var_config['unit_type'] = unit_type
            if time_config:
                s3_path_prefix = var_config.get("s3_path_prefix")
                s3_path_suffix = var_config.get("s3_path_suffix")
                for year in years:
                    file_uri = f"{s3_path_prefix}{year}{s3_path_suffix}"
                    print(f"Processing file URI: {file_uri}")

                    # Open the file once
                    ds = rio.open_rasterio(file_uri)
                    x_chunks = ceil(ds.sizes['x'] / 4000)
                    y_chunks = ceil(ds.sizes['y'] / 4000)

                    for x_chunk in range(x_chunks):
                        for y_chunk in range(y_chunks):
                            chunk = {
                                "x": slice(x_chunk * 4000, (x_chunk + 1) * 4000),
                                "y": slice(y_chunk * 4000, (y_chunk + 1) * 4000)
                            }
                            # Create a delayed task for each chunk
                            task = delayed(process_chunk)(
                                self.session,
                                self.has_time,
                                np.float32 if var_config['unit_type'] == 'float' else np.int16,
                                var_name,
                                group_name,
                                ds,  # Pass the opened dataset
                                chunk
                            )
                            tasks.append(task)
            else:
                s3_path = var_config.get("s3_path")
                print(f"Processing file URI: {s3_path}")

                # Open the file once
                ds = rio.open_rasterio(s3_path)
                x_chunks = ceil(ds.sizes['x'] / 4000)
                y_chunks = ceil(ds.sizes['y'] / 4000)

                for x_chunk in range(x_chunks):
                    for y_chunk in range(y_chunks):
                        chunk = {
                            "x": slice(x_chunk * 4000, (x_chunk + 1) * 4000),
                            "y": slice(y_chunk * 4000, (y_chunk + 1) * 4000)
                        }
                        # Create a delayed task for each chunk
                        task = delayed(process_chunk)(
                            self.session,
                            self.has_time,
                            np.float32 if var_config['unit_type'] == 'float' else np.int16,
                            var_name,
                            group_name,
                            ds,  # Pass the opened dataset
                            chunk
                        )
                        tasks.append(task)

        results = compute(*tasks)

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

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from icechunk.xarray import to_icechunk
from icechunk.distributed import merge_sessions
from arraylake import Client as arraylakeClient
import rioxarray as rio
import numpy as np
from common import ArraylakeDatasetConfig, ORG


def process_annual_dataset_fn(token: str, repo_name: str, has_time: bool, unit_type: str,
                              year: int, var_name: str, group_name: str, file_uri: str):
    """
    Process one annual raster file:
      - reinitializes client and session
      - opens the raster and casts to dtype
      - (if time is enabled) expands dims with the annual timestamp
      - drops unused variables and problematic attributes
      - writes to the repository using icechunk
      - returns the new writable session
    """
    # Reinitialize client and repo in the worker
    client = arraylakeClient(token=token)
    repo = client.get_repo(repo_name)
    new_session = repo.writable_session("main")

    # Open the file and convert to xarray Dataset
    ds = rio.open_rasterio(
        file_uri,
        chunks=(1, 4000, 4000),
        lock=False,
        fill_value=-1,  # Set fill value during read
        masked=True     # Ensure proper handling of NoData values
    ).astype(unit_type).to_dataset(name=var_name)
    ds = ds.squeeze("band", drop=True)

    region = {"x": slice(None), "y": slice(None)}
    if has_time:
        ds = ds.expand_dims(time=[f"{year}-01-01"])
        region = {"time": "auto", "x": slice(None), "y": slice(None)}

    ds = ds.drop_vars(['spatial_ref'])
    for attr in ["add_offset", "scale_factor"]:
        if attr in ds[var_name].attrs:
            del ds[var_name].attrs[attr]

    # (For threaded execution with icechunk, we do not need a context manager)
    new_session.allow_pickling()  # Permanently opt in so the session can be merged later
    to_icechunk(ds.drop_encoding(), new_session,
                group=group_name, region=region)
    return new_session


class ArraylakeRepoPopulator:
    def __init__(self, token: str, dataset_name: str = None, config_dict: dict = None):
        """
        Initialize Populator with either a dataset_name or a configuration dictionary.
        """
        if dataset_name is None and config_dict is None:
            raise ValueError("Must provide either dataset_name or config_dict")
        if dataset_name is not None and config_dict is not None:
            raise ValueError(
                "Cannot provide both dataset_name and config_dict")

        # Load configuration
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

        # Initialize client and repo
        self.client = arraylakeClient(token=token)
        self.repo = self.client.get_repo(self.repo_name)
        self.session = self.repo.writable_session("main")
        self.groups = self.config.get('groups', {})

        # Handle dimensions
        self.dims = self.config.get('dim', ['x', 'y'])
        self.has_time = 'time' in self.dims

    def populate_group(self, group_name: str) -> None:
        """
        For each variable (except 'time') in the group's configuration, compute the annual file URI and process
        concurrently using ThreadPoolExecutor. Merge all returned sessions and commit the changes.
        """
        if group_name not in self.config['groups']:
            raise ValueError(f"Group {group_name} not found in config.")
        group_config = self.config['groups'][group_name]
        time_config = group_config.get("time", None)
        if time_config:
            start_date = pd.Timestamp(time_config.get("start", "2000-01-01"))
            end_date = pd.Timestamp(time_config.get("end", "2024-01-01"))
            freq = time_config.get("freq", "YS")
            years = pd.date_range(start_date, end_date,
                                  freq=freq).year.tolist()
        futures = []
        with ThreadPoolExecutor() as executor:
            # Loop through variables (skip 'time')
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
                    s3_path = var_config.get("s3_path")
                    future = executor.submit(
                        process_annual_dataset_fn,
                        self.token,
                        self.repo_name,
                        self.has_time,
                        np.float32 if var_config['unit_type'] == 'float' else np.int16,
                        year,
                        var_name,
                        group_name,
                        s3_path,
                    )
                    futures.append(future)
            # Wait for all tasks to complete.
            results = [future.result() for future in futures]

        # Merge sessions returned from each task.
        self.session = merge_sessions(self.session, *results)
        print(f"Populated group {group_name} with task results: {results}")
        self.session.commit(
            f"Populated group {group_name} with task results: {results}")

    def populate_all_groups(self) -> None:
        """
        Iterate over every group in the config and populate each.
        """
        for group in self.config.get("groups", {}).keys():
            print(f"Populating group: {group}")
            self.populate_group(group)

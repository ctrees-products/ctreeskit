# Standard library imports
import json
from typing import Optional, Dict, Any

# Third-party library imports
import s3fs
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da
import rioxarray as rio
import pyproj
from shapely.geometry import shape
from shapely.ops import transform

# Local application/library specific imports
from arraylake import Client as arraylakeClient
from common import ArraylakeDatasetConfig


class ArraylakeRepoInitializer:
    def __init__(
        self,
        token: str,
        dataset_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        geojson_path: Optional[str] = None,
    ):
        """Initialize Initializer with either dataset_name or config dictionary.

        Args:
            token: API token for authentication
            dataset_name: Name of dataset to load config for
            config_dict: Direct configuration dictionary
            geojson_path: Optional path to GeoJSON for spatial subsetting
        """
        print("loading config")
        if dataset_name is None and config_dict is None:
            raise ValueError("Must provide either dataset_name or config_dict")

        if dataset_name is not None and config_dict is not None:
            raise ValueError(
                "Cannot provide both dataset_name and config_dict")

        # Load config from dataset name or use provided dict
        if dataset_name is not None:
            config_loader = ArraylakeDatasetConfig(dataset_name)
            self.config = config_loader._config
        else:
            self.config = config_dict
        self.dataset_name = self.config['dataset_name']
        self.repo_name = self.config.get("repo", f"{self.config['organization']/{self.dataset_name}")
        self.crs = self.config.get('crs', 'EPSG:4326')

        # Initialize client and repo
        self.client = arraylakeClient(token=token)
        self.repo = self.client.get_repo(self.repo_name)
        self.session = self.repo.writable_session("main")
        self.groups = self.config.get('groups', {})

        # Handle dimensions
        self.dims = self.config.get('dim', ['x', 'y'])
        self.has_time = 'time' in self.dims

        # Process bbox if geojson provided
        self.bbox = None
        if geojson_path:
            self.bbox = self._process_geometry(geojson_path)

    def _process_geometry(self, geojson_path: str) -> tuple:
        """Process GeoJSON geometry and ensure CRS matches dataset.

        Args:
            geojson_path: Path to GeoJSON file. This can be a local file path or an S3 URI.

        Returns:
            tuple: (minx, miny, maxx, maxy) in dataset CRS.
        """
        # Read GeoJSON using s3fs if the path is an S3 URI.
        if geojson_path.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(geojson_path, 'r') as f:
                geojson_dict = json.load(f)
        else:
            with open(geojson_path, 'r') as f:
                geojson_dict = json.load(f)

        # Get geometry and its CRS.
        geometry = shape(geojson_dict['features'][0]['geometry'])
        geom_crs = geojson_dict.get('crs', {}).get(
            'properties', {}).get('name', 'EPSG:4326')

        # Transform if CRS differs.
        if geom_crs != self.crs:
            source_crs = pyproj.CRS(geom_crs)
            target_crs = pyproj.CRS(self.crs)
            project = pyproj.Transformer.from_crs(
                source_crs,
                target_crs,
                always_xy=True
            ).transform
            geometry = transform(project, geometry)

        return geometry.bounds

    def initialize_all_groups(self) -> None:
        """Initialize all groups defined in the configuration."""
        groups = self.config.get("groups", {})
        if not groups:
            raise ValueError("No groups defined in the configuration.")
        for group_name in groups.keys():
            print(f"Initializing group: {group_name}")
            self.initialize_group(group_name)

    def initialize_group(self, group_name: str) -> None:
        """Initialize a group with all its variables from config.

        Args:
            group_name: Name of the group in config
        """
        if group_name not in self.config['groups']:
            raise ValueError(f"Group {group_name} not found")

        group_config = self.config['groups'][group_name]

        # Filter out time config and get variable names
        variables = {k: v for k, v in group_config.items() if k != 'time'}

        if not variables:
            raise ValueError(f"No variables found in group {group_name}")

        # Collect configurations and base rasters for all variables
        base_rasters = {}
        var_configs = {}

        for var_name, var_config in variables.items():
            var_configs[var_name] = var_config

            # Get base raster path for each variable
            if 's3_path' in var_config:
                base_rasters[var_name] = var_config['s3_path']
            else:
                # Handle time series data
                time_config = group_config.get('time', {})
                start_date = pd.Timestamp(
                    time_config.get('start', '2000-01-01'))
                base_rasters[var_name] = f"{var_config['s3_path_prefix']}{start_date.year}{var_config['s3_path_suffix']}"

        # Create schema with all variables
        ds = self.create_schema(group_name, base_rasters, var_configs)
        # Write to repo
        chunks = {"time": 1, "y": 2000, "x": 2000} if self.has_time else {
            "y": 2000, "x": 2000}
        encoding = self._construct_chunks_encoding(ds, chunks)
        ds = ds.chunk(chunks)

        if group_name != "root":
            ds.to_zarr(
                self.session.store,
                group=group_name,
                mode="w",
                encoding=encoding,
                compute=False
            )
        else:
            ds.to_zarr(
                self.session.store,
                mode="w",
                encoding=encoding,
                compute=False
            )
        print(f"initialized group: {group_name}")
        self.session.commit(f"initialized group: {group_name}")

    def create_schema(
        self,
        group_name: str,
        base_rasters: Dict[str, str],
        var_configs: Dict[str, Dict[str, Any]]
    ) -> xr.Dataset:
        """Create schema based on config and base raster.

        Args:
            group_name: Name of the group in config
            base_rasters: Dict mapping variable names to base raster paths
            var_configs: Dict mapping variable names to their configurations
        """
        # Read template raster from first variable to establish coordinates
        first_var = list(base_rasters.keys())[0]
        with rio.open_rasterio(base_rasters[first_var],  chunks=(1, 4000, 4000), lock=False) as src:
            template = src.isel(band=0).to_dataset(name=first_var)
            if template.rio.crs != self.crs:
                template = template.rio.reproject(self.crs)
            resolution = template.rio.resolution()[0]

        # Get coordinates
        if self.bbox is not None and not var_configs[first_var].get('is_mosaiced', False):
            min_x, min_y, max_x, max_y = self.bbox
            x = np.arange(min_x, max_x, resolution)
            y = np.arange(min_y, max_y, resolution)
        else:
            x = template.x
            y = template.y

        # Create time coordinates if needed
        coords = {"y": y, "x": x}
        if self.has_time:
            time_config = self.config['groups'][group_name].get('time', {})
            time_range = pd.date_range(
                start=time_config.get('start'),
                end=time_config.get('end'),
                freq=time_config.get('freq', 'YS')
            )
            coords["time"] = time_range
        # Create empty dataset with all variables, but use lazy dask arrays instead of np.zeros
        data_vars = {}
        # Determine shape from coordinate lengths in self.dims
        shape = tuple(len(coords[dim]) for dim in self.dims)
        # Prepare chunk sizes based on your chunking scheme. For example:
        default_chunks = [2000 if dim in [
            'y', 'x'] else 1 for dim in self.dims]
        for var_name, var_config in var_configs.items():
            # Use int8 if that’s your choice for non-float variables
            dtype = np.float32 if var_config['unit_type'] == 'float' else np.int16

            data_vars[var_name] = (
                self.dims,
                da.zeros(shape, dtype=dtype,
                         chunks=default_chunks, fill_value=-1)
            )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )

        return ArraylakeDatasetConfig().add_cf_metadata(ds, self.config)

    def _construct_chunks_encoding(self, ds: xr.Dataset, chunks: dict) -> dict:
        """Construct encoding dictionary for zarr storage

        Args:
            ds: xarray Dataset to be chunked
            chunks: Dictionary of chunk sizes for each dimension

        Returns:
            dict: Encoding dictionary for zarr storage with chunk sizes
        """
        return {
            name: {"chunks": tuple(
                chunks.get(dim, var.sizes[dim]) for dim in var.dims
            )
            }
            for name, var in ds.data_vars.items()
        }

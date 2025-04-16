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
from .common import ArraylakeDatasetConfig


class ArraylakeRepoInitializer:
    """
    A class for initializing an Arraylake repository from a configuration.

    This class loads configuration information (either from S3 via a dataset name
    or directly from a dictionary) and sets up an Arraylake repository accordingly.
    It also handles spatial subsetting using geometry from a GeoJSON file to define
    a bounding box. Additionally, it creates an xarray Dataset schema for each group
    defined in the configuration and writes the dataset (e.g., to Zarr).

    Attributes
    ----------
    config : Dict[str, Any]
        Dataset configuration dictionary.
    dataset_name : str
        The name of the dataset as specified in the configuration.
    repo_name : str
        Repository name built from organization and dataset name.
    crs : str
        Coordinate Reference System for spatial operations (default is 'EPSG:4326').
    client : arraylakeClient
        Arraylake API client used for communicating with the Arraylake service.
    repo : Repository object
        Repository object retrieved from the Arraylake client.
    session : Session object
        A writable session for updating the repository.
    groups : Dict[str, Any]
        Configuration for each group (logical grouping of variables).
    dims : list
        List of dataset dimension names (e.g., ['x', 'y'] or including 'time').
    has_time : bool
        Flag indicating whether the dataset includes a time dimension.
    bbox : tuple or None
        Spatial bounding box (minx, miny, maxx, maxy) for subsetting, if provided.
    """

    def __init__(
        self,
        token: str,
        dataset_name: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        geojson_path: Optional[str] = None,
    ):
        """
        Initialize the ArraylakeRepoInitializer with necessary configuration.

        This constructor requires either a dataset_name to load configuration from S3 
        or a provided configuration dictionary. Additionally, an optional geojson_path 
        can be provided for spatial subsetting. The method initializes the Arraylake 
        client, retrieves the repository, and sets up dimensions and groups as defined in 
        the configuration.

        Parameters
        ----------
        token : str
            API token for authentication with the Arraylake service.
        dataset_name : Optional[str]
            Name of the dataset to load configuration for from S3. Mutually exclusive with config_dict.
        config_dict : Optional[Dict[str, Any]]
            A configuration dictionary provided directly. Mutually exclusive with dataset_name.
        geojson_path : Optional[str]
            Path (local or S3 URI) to a GeoJSON file for spatial subsetting.

        Raises
        ------
        ValueError
            If neither dataset_name nor config_dict is provided or if both are provided.
        """
        print("loading config")
        if dataset_name is None and config_dict is None:
            raise ValueError("Must provide either dataset_name or config_dict")
        if dataset_name is not None and config_dict is not None:
            raise ValueError(
                "Cannot provide both dataset_name and config_dict")

        # Load configuration from S3 if dataset_name is provided, otherwise use the given dictionary.
        if dataset_name is not None:
            config_loader = ArraylakeDatasetConfig(dataset_name)
            self.config = config_loader._config
        else:
            self.config = config_dict

        # Set key properties from configuration.
        self.dataset_name = self.config['dataset_name']
        self.repo_name = self.config.get(
            "repo", f"{self.config['organization']}/{self.dataset_name}")
        self.crs = self.config.get('crs', 'EPSG:4326')

        # Initialize Arraylake client, get repository, and open a writable session.
        self.client = arraylakeClient(token=token)
        self.repo = self.client.get_repo(self.repo_name)
        self.session = self.repo.writable_session("main")
        self.groups = self.config.get('groups', {})

        # Setup dimensions and flag for time dimension.
        self.dims = self.config.get('dim', ['x', 'y'])
        self.has_time = 'time' in self.dims

        # If a GeoJSON path is provided, process it to create a bounding box.
        self.bbox = None
        if geojson_path:
            self.bbox = self._process_geometry(geojson_path)

    def _process_geometry(self, geojson_path: str) -> tuple:
        """
        Process GeoJSON geometry file and ensure its CRS matches the dataset's CRS.

        Reads the provided GeoJSON (local file or S3 URI) and extracts the geometry. 
        If the CRS of the geometry differs from the dataset's CRS, it reprojects the geometry.

        Parameters
        ----------
        geojson_path : str
            Path to the GeoJSON file (local file path or S3 URI).

        Returns
        -------
        tuple
            A tuple (minx, miny, maxx, maxy) corresponding to the geometry bounds in the dataset's CRS.
        """
        # Open GeoJSON using s3fs if the URI indicates S3; else use the local filesystem.
        if geojson_path.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(geojson_path, 'r') as f:
                geojson_dict = json.load(f)
        else:
            with open(geojson_path, 'r') as f:
                geojson_dict = json.load(f)

        # Extract the geometry and its CRS
        geometry = shape(geojson_dict['features'][0]['geometry'])
        geom_crs = geojson_dict.get('crs', {}).get(
            'properties', {}).get('name', 'EPSG:4326')

        # If the geometry's CRS differs from the dataset's, reproject it.
        if geom_crs != self.crs:
            source_crs = pyproj.CRS(geom_crs)
            target_crs = pyproj.CRS(self.crs)
            project = pyproj.Transformer.from_crs(
                source_crs, target_crs, always_xy=True).transform
            geometry = transform(project, geometry)

        return geometry.bounds

    def initialize_all_groups(self, fill_value=-1) -> None:
        """
        Initialize all variable groups defined in the configuration.

        Iterates over each group in the 'groups' section of the configuration, 
        invoking initialize_group() for each. Raises an error if no groups are defined.

        Raises
        ------
        ValueError
            If no groups are defined in the configuration.
        """
        groups = self.config.get("groups", {})
        if not groups:
            raise ValueError("No groups defined in the configuration.")
        for group_name in groups.keys():
            print(f"Initializing group: {group_name}")
            self.initialize_group(group_name, fill_value)

    def initialize_group(self, group_name: str, fill_value=-1) -> None:
        """
        Initialize a specific group from the configuration.

        This method processes the configuration for the given group, extracts variable-specific 
        settings, determines base raster paths, creates an xarray Dataset schema, and writes 
        the dataset to a Zarr store. The dataset is chunked based on the supplied dimensions.

        Parameters
        ----------
        group_name : str
            The name of the group to initialize.

        Raises
        ------
        ValueError
            If the group is not found in the configuration or if no variables are defined.
        """
        if group_name not in self.config['groups']:
            raise ValueError(f"Group {group_name} not found")

        group_config = self.config['groups'][group_name]

        # Exclude 'time' config and get remaining variable configurations.
        variables = {k: v for k, v in group_config.items() if k != 'time'}
        if not variables:
            raise ValueError(f"No variables found in group {group_name}")

        # Collect base rasters and variable configurations.
        base_rasters = {}
        var_configs = {}

        for var_name, var_config in variables.items():
            var_configs[var_name] = var_config

            # Use s3_path if defined; otherwise, construct path based on time series data.
            if 's3_path' in var_config:
                base_rasters[var_name] = var_config['s3_path']
            else:
                time_config = group_config.get('time', {})
                start_date = pd.Timestamp(
                    time_config.get('start', '2000-01-01'))
                base_rasters[var_name] = f"{var_config['s3_path_prefix']}{start_date.year}{var_config['s3_path_suffix']}"

        # Create dataset schema for the group.
        ds = self.create_schema(group_name, base_rasters,
                                var_configs, fill_value)

        # Determine chunk sizes for dimensions.
        chunks = {"time": 1, "y": 2000, "x": 2000} if self.has_time else {
            "y": 2000, "x": 2000}
        encoding = self._construct_chunks_encoding(
            ds, chunks, fill_value)
        ds = ds.chunk(chunks)

        # Write dataset to Zarr storage.
        if group_name != "root":
            ds.drop_encoding().to_zarr(self.session.store, group=group_name,
                                       mode="w", encoding=encoding, compute=False)
        else:
            ds.drop_encoding().to_zarr(self.session.store, mode="w",
                                       encoding=encoding, compute=False)
        print(f"initialized group: {group_name}")

        # Commit changes to the repository session.
        self.session.commit(f"initialized group: {group_name}")

    def create_schema(
        self,
        group_name: str,
        base_rasters: Dict[str, str],
        var_configs: Dict[str, Dict[str, Any]],
        fill_value=-1
    ) -> xr.Dataset:
        """
        Create an xarray Dataset schema based on the configuration and base rasters.

        This method establishes coordinates (x, y, and optionally time) and creates data variables 
        using lazy Dask arrays (with -1 filled in). The resolution is determined from the first 
        variable's raster. CF metadata is then added to the dataset.

        Parameters
        ----------
        group_name : str
            Name of the group configuration.
        base_rasters : Dict[str, str]
            A dictionary mapping variable names to their base raster file paths.
        var_configs : Dict[str, Dict[str, Any]]
            A dictionary mapping variable names to their individual configurations.

        Returns
        -------
        xr.Dataset
            An xarray Dataset with proper coordinates, variables, and CF-compliant metadata.
        """
        # Read a template raster to establish coordinate reference and resolution.
        first_var = list(base_rasters.keys())[0]
        with rio.open_rasterio(base_rasters[first_var], chunks=(1, 4000, 4000), lock=False) as src:
            template = src.isel(band=0).to_dataset(name=first_var)
            if template.rio.crs != self.crs:
                template = template.rio.reproject(self.crs)
            resolution = template.rio.resolution()[0]

        # Determine coordinates based on bounding box (if provided and not mosaiced) or template.
        if self.bbox is not None and not var_configs[first_var].get('is_mosaiced', False):
            min_x, min_y, max_x, max_y = self.bbox
            x = np.arange(min_x, max_x, resolution)
            y = np.arange(min_y, max_y, resolution)
        else:
            x = template.x
            y = template.y

        # Create coordinates dictionary, adding time if applicable.
        coords = {"y": y, "x": x}
        if self.has_time:
            time_config = self.config['groups'][group_name].get('time', {})
            time_range = pd.date_range(
                start=time_config.get('start'),
                end=time_config.get('end'),
                freq=time_config.get('freq', 'YS')
            )
            coords["time"] = time_range

        # Build data variables using the dimensions specified in self.dims.
        data_vars = {}
        shape = tuple(len(coords[dim]) for dim in self.dims)
        default_chunks = [2000 if dim in [
            'y', 'x'] else 1 for dim in self.dims]
        for var_name, var_config in var_configs.items():
            dtype = np.float32 if var_config['unit_type'] == 'float' else np.int16
            data_vars[var_name] = (
                self.dims,
                da.ones(shape,
                        dtype=dtype, chunks=default_chunks)
            )
        ds = xr.Dataset(data_vars=data_vars, coords=coords)
        # Add CF metadata to dataset using ArraylakeDatasetConfig helper.
        return ArraylakeDatasetConfig().add_cf_metadata(ds, self.config)

    def _construct_chunks_encoding(self, ds: xr.Dataset, chunks: dict, fill_value=-1) -> dict:
        """
        Construct an encoding dictionary for writing an xarray Dataset to Zarr storage.

        The encoding dictionary maps each variable to its specific chunk sizes, which are 
        determined based on the provided chunks dictionary and the variable dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to be written.
        chunks : dict
            Dictionary specifying chunk sizes for each dimension.

        Returns
        -------
        dict
            An encoding dictionary suitable for use with ds.to_zarr().
        """
        return {
            name: {
                "chunks": tuple(chunks.get(dim, var.sizes[dim]) for dim in var.dims),
                "fill_value": fill_value  # Add the fill_value to the encoding
            }
            for name, var in ds.data_vars.items()
        }

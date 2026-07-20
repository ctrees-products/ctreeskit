# Standard library imports
import json
from typing import Optional, Dict, Any

# Third-party library imports
import numpy as np
import xarray as xr
import pyproj

from .._s3 import s3_client


class ArraylakeDatasetConfig:
    """
    Handles dataset configuration loading and validation from a config file.

    This class can either load a configuration from an S3 bucket using a dataset name,
    or be provided a configuration dictionary directly. It also provides helper properties
    and methods to extract and add standardized metadata to an xarray Dataset based on
    configuration information.

    Attributes
    ----------
    _config : Dict[str, Any]
        The internal configuration dictionary loaded from S3 or passed in directly.
    organization : str
        The organization name associated with datasets.
    bucket : str
        The S3 bucket where dataset configuration files are stored.
    config_prefix : str
        The path prefix in the S3 bucket for locating configuration files.
    """

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        organization: str = "",
        bucket: str = "arraylake-datasets",
        config_prefix: str = "configs/",
    ):
        """
        Initialize dataset configuration handler.

        This constructor will attempt to load a configuration file (in JSON format)
        from the specified S3 bucket and prefix if a dataset_name is provided.
        Otherwise, the configuration can be supplied directly via _config.

        Parameters
        ----------
        dataset_name : Optional[str]
            Optional name of a specific dataset to load configuration for.
        organization : str
            Organization name to be included in repository naming if not in config.
        bucket : str
            S3 bucket containing configuration files (default: "arraylake-datasets").
        config_prefix : str
            Path prefix inside the S3 bucket for configuration files (default: "configs/").
        """
        self.bucket = bucket
        self.config_prefix = config_prefix
        self._config: Dict[str, Any] = {}

        # If dataset_name is provided, load the configuration from S3.
        if dataset_name:
            self.load_config(dataset_name)

        # Set the organization, using the value in config if available.
        self.organization = self._config.get('organization', organization)

    def load_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load configuration for a specific dataset from S3.

        Constructs the S3 key from the config_prefix and dataset_name, then retrieves
        and decodes the JSON configuration.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset configuration to load (without the .json extension).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing dataset configuration data.

        Raises
        ------
        ValueError
            If the configuration could not be loaded (e.g., due to a ClientError).
        ImportError
            If the optional ``s3`` extra is not installed.
        """
        client = s3_client()
        key = f"{self.config_prefix}{dataset_name}.json"
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
            self._config = json.load(response["Body"])
            return self._config
        except Exception as e:
            raise ValueError(
                f"Could not load config for {dataset_name}: {e}") from e

    def list_datasets(self) -> list:
        """
        List all available dataset configurations from the S3 bucket.

        Retrieves all objects under the configured prefix and extracts dataset names
        from the filenames.

        Returns
        -------
        list
            List of dataset names (strings) available in the specified S3 bucket/prefix.

        Raises
        ------
        ValueError
            If there is an error during listing (e.g., S3 access issues).
        ImportError
            If the optional ``s3`` extra is not installed.
        """
        client = s3_client()
        try:
            paginator = client.get_paginator("list_objects_v2")
            return [
                obj["Key"].split("/")[-1].removesuffix(".json")
                for page in paginator.paginate(
                    Bucket=self.bucket, Prefix=self.config_prefix)
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".json")
            ]
        except Exception as e:
            raise ValueError(f"Could not list datasets: {e}") from e

    @property
    def dataset_name(self) -> Optional[str]:
        """
        Retrieve the dataset name from the configuration.

        Returns
        -------
        Optional[str]
            The dataset name as specified in the configuration, if available.
        """
        return self._config.get('dataset_name')

    @property
    def repo_name(self) -> Optional[str]:
        """
        Retrieve the repository name from the configuration.

        Defaults to a string combining the organization and dataset name if a specific repo name
        is not provided in the configuration.

        Returns
        -------
        Optional[str]
            The repository name.
        """
        return self._config.get('repo', f"{self.organization}/{self.dataset_name}")

    @property
    def long_name(self) -> Optional[str]:
        """
        Retrieve a long descriptive name for the dataset from the configuration.

        Returns
        -------
        Optional[str]
            Long descriptive name if provided; otherwise, None.
        """
        return self._config.get('long_name')

    @property
    def crs(self) -> str:
        """
        Retrieve the coordinate reference system for the dataset.

        Returns
        -------
        str
            The CRS (default is 'EPSG:4326') as specified in the configuration.
        """
        return self._config.get('crs', 'EPSG:4326')

    @property
    def dimensions(self) -> list:
        """
        Retrieve the dimensions defined for the dataset.

        Returns
        -------
        list
            A list of dimension names (default is ['x', 'y']).
        """
        return self._config.get('dim', ['x', 'y'])

    @property
    def has_time(self) -> bool:
        """
        Determine if the dataset includes a time dimension.

        Returns
        -------
        bool
            True if 'time' is present in the dimensions, False otherwise.
        """
        return 'time' in self.dimensions

    def get_group_config(self, group_name: str) -> Dict[str, Any]:
        """
        Retrieve configuration for a specific group within the dataset.

        This method searches the configuration's 'groups' dictionary for the provided
        group name. If not found, a ValueError is raised.

        Parameters
        ----------
        group_name : str
            The name of the group to retrieve.

        Returns
        -------
        Dict[str, Any]
            The configuration dictionary for the specified group.

        Raises
        ------
        ValueError
            If the group name is not found in the configuration.
        """
        groups = self._config.get('groups', {})
        if group_name not in groups:
            raise ValueError(f"Group {group_name} not found in config")
        return groups[group_name]

    def add_cf_metadata(
        self,
        ds: xr.Dataset,
        config: Optional[Dict[str, Any]] = None,
        crs: Optional[Any] = None,
    ) -> xr.Dataset:
        """
        Add CF (Climate and Forecast) compliant metadata to an xarray Dataset.

        Updates coordinate variables (x, y, and optionally time) with standard metadata.
        Then iterates over each data variable in the dataset, searching for matching
        variable-specific metadata in the configuration groups. Attribution is added for:

        - Classification variables with a "values" mapping.
        - Variables with a defined "unit_name" and associated properties.

        The x/y coordinate metadata is derived from the dataset's *actual* CRS rather
        than hardcoded to longitude/latitude: geographic CRSs get longitude/latitude
        (``degrees_east``/``degrees_north``), while projected CRSs get the CF
        ``projection_x_coordinate``/``projection_y_coordinate`` standard names with the
        CRS's own linear unit. This follows rioxarray's CF grid-mapping convention; when
        a ``spatial_ref`` coordinate is present each data variable is linked to it via a
        ``grid_mapping`` attribute.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset to be augmented with metadata.
        config : Optional[Dict[str, Any]]
            An optional configuration dictionary. If provided, it will be used
            instead of the internally stored configuration (self._config).
        crs : Optional[Any]
            An optional CRS (anything ``pyproj.CRS.from_user_input`` accepts). Takes
            precedence when resolving coordinate metadata. If omitted, the CRS is
            inferred from the dataset's ``spatial_ref`` coordinate, then the config's
            ``crs`` key, then WGS84.
        Returns
        -------
        xr.Dataset
            The input dataset updated with CF-compliant metadata.
        """
        # Use the provided config if available; otherwise, use the internal config.
        cfg = config if config is not None else self._config

        # Resolve the CRS so coordinate metadata reflects the real projection instead
        # of assuming lon/lat, then apply CRS-appropriate x/y attributes.
        crs_obj = self._resolve_crs(ds, cfg, crs)
        x_attrs, y_attrs = self._coordinate_cf_attrs(crs_obj)
        if "x" in ds.coords:
            ds.x.attrs.update(x_attrs)
        if "y" in ds.coords:
            ds.y.attrs.update(y_attrs)

        # Optionally update the time coordinate if present.
        if 'time' in ds.coords:
            ds.time.attrs.update({
                "standard_name": "time",
                "long_name": "time",
                "axis": "T",
            })

        has_grid_mapping = 'spatial_ref' in ds.coords

        # Process metadata for each data variable using the configuration groups.
        groups = cfg.get('groups', {})
        for var_name in ds.data_vars:
            var_config = None
            # Search for a matching variable configuration in all groups.
            for group in groups.values():
                if var_name in group:
                    var_config = group[var_name]
                    break

            attrs: Dict[str, Any] = {}
            # Link the variable to its grid mapping (rioxarray CF convention).
            if has_grid_mapping:
                attrs["grid_mapping"] = "spatial_ref"

            if var_config:
                # For classification variables with a "values" mapping.
                if 'values' in var_config:
                    flag_dict = var_config['values']
                    sorted_items = sorted(
                        flag_dict.items(), key=lambda x: int(x[0]))
                    # Keep flag_values in the same dtype as the variable, as CF requires.
                    var_dtype = ds[var_name].dtype
                    flag_dtype = var_dtype if np.issubdtype(
                        var_dtype, np.integer) else np.int16
                    flag_values = np.array(
                        [int(k) for k, _ in sorted_items], dtype=flag_dtype)
                    flag_meanings = ' '.join(v.replace(' ', '_')
                                             for _, v in sorted_items)
                    classification_type = var_config.get(
                        'classification_type', 'classification')
                    # CF-1.8 flag variable (§3.5): flag_values + flag_meanings
                    # make it self-describing. We intentionally omit `units`
                    # (flags are categorical, not dimensional — CF's own flag
                    # examples carry no units) and `standard_name` (there is no
                    # CF standard name for a land-cover classification, and the
                    # `status_flag` modifier form is for QC flags about another
                    # variable, not for categorical data itself). `long_name`
                    # satisfies CF's "at least one of standard_name/long_name".
                    attrs.update({
                        "long_name": classification_type,
                        "flag_values": flag_values,
                        "flag_meanings": flag_meanings,
                        "classification_type": classification_type,
                    })
                # For variables with a defined unit.
                elif 'unit_name' in var_config:
                    attrs.update({
                        "units": var_config['unit_name'],
                    })

            # Update the variable's attributes if any were determined.
            if attrs:
                ds[var_name].attrs.update(attrs)

        # Set global attribute for CF Conventions.
        ds.attrs.update({
            "Conventions": "CF-1.8"
        })
        return ds

    @staticmethod
    def _resolve_crs(ds: xr.Dataset, cfg: Dict[str, Any], crs: Optional[Any]) -> "pyproj.CRS":
        """
        Determine the CRS to use for coordinate metadata.

        Precedence: an explicit ``crs`` argument, then the dataset's rioxarray-style
        ``spatial_ref`` coordinate (``crs_wkt``/``spatial_ref`` attribute), then the
        config's ``crs`` key, then WGS84.
        """
        if crs is not None:
            return pyproj.CRS.from_user_input(crs)
        if 'spatial_ref' in getattr(ds, 'coords', {}):
            sr_attrs = ds['spatial_ref'].attrs
            wkt = sr_attrs.get('crs_wkt') or sr_attrs.get('spatial_ref')
            if wkt:
                try:
                    return pyproj.CRS.from_user_input(wkt)
                except Exception:
                    pass
        return pyproj.CRS.from_user_input(cfg.get('crs', 'EPSG:4326'))

    @staticmethod
    def _coordinate_cf_attrs(crs_obj: "pyproj.CRS") -> tuple:
        """
        Build CF-compliant attribute dicts for the x and y coordinates given a CRS.

        Geographic CRSs map to longitude/latitude; projected CRSs map to CF
        ``projection_x_coordinate``/``projection_y_coordinate`` with the CRS's own
        linear unit (defaulting to metre).
        """
        if crs_obj.is_geographic:
            x_attrs = {"standard_name": "longitude", "long_name": "longitude",
                       "units": "degrees_east", "axis": "X"}
            y_attrs = {"standard_name": "latitude", "long_name": "latitude",
                       "units": "degrees_north", "axis": "Y"}
            return x_attrs, y_attrs

        unit = "metre"
        try:
            for axis in crs_obj.axis_info:
                if axis.unit_name:
                    unit = axis.unit_name
                    break
        except Exception:
            pass
        x_attrs = {"standard_name": "projection_x_coordinate",
                   "long_name": "x coordinate of projection",
                   "units": unit, "axis": "X"}
        y_attrs = {"standard_name": "projection_y_coordinate",
                   "long_name": "y coordinate of projection",
                   "units": unit, "axis": "Y"}
        return x_attrs, y_attrs

# Standard library imports
import json
from typing import Optional, Dict, Any

# Third-party library imports
import boto3
from botocore.exceptions import ClientError
import numpy as np
import xarray as xr


class ArraylakeDatasetConfig:
    """Handles dataset configuration loading and validation."""

    def __init__(
        self,
        dataset_name: Optional[str] = None,
        organization: str = "",
        bucket: str = "arraylake-datasets",
        config_prefix: str = "configs/",

    ):
        """Initialize dataset configuration handler.

        Args:
            dataset_name: Optional name of specific dataset to load
            bucket: S3 bucket containing configs (default: arraylake-datasets)
            config_prefix: Path prefix for config files (default: configs/)
        """
        self.s3 = boto3.client('s3')
        self.bucket = bucket
        self.config_prefix = config_prefix
        self._config: Dict[str, Any] = {}

        if dataset_name:
            self.load_config(dataset_name)

        self.organization = self._config.get(
            'organization', organization)

    def load_config(self, dataset_name: str) -> Dict[str, Any]:
        """Load configuration for a specific dataset.

        Args:
            dataset_name: Name of the dataset to load

        Returns:
            Dict containing dataset configuration
        """
        key = f"{self.config_prefix}{dataset_name}.json"
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            self._config = json.loads(response['Body'].read().decode('utf-8'))
            return self._config
        except ClientError as e:
            raise ValueError(f"Could not load config for {dataset_name}: {e}")

    def list_datasets(self) -> list:
        """List all available dataset configurations.

        Returns:
            List of dataset names
        """
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.config_prefix
            )
            return [
                obj['Key'].split('/')[-1].replace('.json', '')
                for obj in response.get('Contents', [])
                if obj['Key'].endswith('.json')
            ]
        except ClientError as e:
            raise ValueError(f"Could not list datasets: {e}")

    @property
    def dataset_name(self) -> Optional[str]:
        """Get dataset name from config."""
        return self._config.get('dataset_name')

    @property
    def repo_name(self) -> Optional[str]:
        """Get repository name from config."""
        return self._config.get('repo', f"{self.organization}/{self.dataset_name}")

    @property
    def long_name(self) -> Optional[str]:
        """Get long descriptive name from config."""
        return self._config.get('long_name')

    @property
    def crs(self) -> str:
        """Get CRS from config."""
        return self._config.get('crs', 'EPSG:4326')

    @property
    def dimensions(self) -> list:
        """Get dimensions from config."""
        return self._config.get('dim', ['x', 'y'])

    @property
    def has_time(self) -> bool:
        """Check if dataset has time dimension."""
        return 'time' in self.dimensions

    def get_group_config(self, group_name: str) -> Dict[str, Any]:
        """Get configuration for a specific group.

        Args:
            group_name: Name of the group to retrieve

        Returns:
            Dict containing group configuration
        """
        groups = self._config.get('groups', {})
        if group_name not in groups:
            raise ValueError(f"Group {group_name} not found in config")
        return groups[group_name]

    def add_cf_metadata(self, ds: xr.Dataset, config: Optional[Dict[str, Any]] = None) -> xr.Dataset:
        """
        Add CF (Climate and Forecast) compliant metadata to the provided xarray Dataset.

        This function updates coordinate variables (x, y, and optionally time) with
        standard metadata and then iterates over the dataset's data variables to add
        variable-specific metadata based on configuration group definitions.

        The metadata extraction can either use the internal configuration (self._config)
        or an external config dictionary that is passed in.

        Parameters
        ----------
        ds : xr.Dataset
            The xarray Dataset to which metadata will be added.
        config : Optional[Dict[str, Any]]
            An optional configuration dictionary to use instead of self._config.

        Returns
        -------
        xr.Dataset
            The updated Dataset with CF-compliant metadata.
        """
        # Use the provided config if available, otherwise fall back to the internal one.
        cfg = config if config is not None else self._config

        # Update coordinate metadata
        ds.x.attrs.update({
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east"
        })

        ds.y.attrs.update({
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north"
        })

        if 'time' in ds:
            ds.time.attrs.update({
                "standard_name": "time",
                "long_name": "time"
            })

        # Process each data variable metadata using configuration groups
        groups = cfg.get('groups', {})
        for var_name in ds.data_vars:
            var_config = None
            # Search for variable configuration in each group.
            for group in groups.values():
                if var_name in group:
                    var_config = group[var_name]
                    break

            # Skip if no configuration is found.
            if not var_config:
                continue

            attrs = {}
            # If the variable is a classification variable with a "values" mapping:
            if 'values' in var_config:
                flag_dict = var_config['values']
                sorted_items = sorted(
                    flag_dict.items(), key=lambda x: int(x[0]))
                flag_values = np.array(
                    [int(k) for k, v in sorted_items], dtype=np.int16)
                flag_meanings = ' '.join(v.replace(' ', '_')
                                         for k, v in sorted_items)
                classification_type = var_config.get(
                    'classification_type', 'classification')
                attrs.update({
                    "standard_name": "classification",
                    "long_name": classification_type,
                    "flag_values": flag_values,
                    "flag_meanings": flag_meanings,
                    "units": "class",
                    "classification_type": classification_type,
                    "_FillValue": np.int16(-1)
                })
            # Otherwise, if the variable has a unit name defined:
            elif 'unit_name' in var_config:
                attrs.update({
                    "units": var_config['unit_name'],
                    "_FillValue": np.int16(-1) if var_config.get('unit_type') == 'int' else np.float32(-1)
                })

            # Update variable attributes if any attributes were determined.
            if attrs:
                ds[var_name].attrs.update(attrs)

        # Update global dataset attributes.
        ds.attrs.update({
            "Conventions": "CF-1.8"
        })
        return ds

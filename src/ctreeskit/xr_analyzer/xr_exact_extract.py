from typing import Union, Optional
import json
import xarray as xr
import numpy as np
from shapely.geometry import box, shape
from common import Units, AreaUnit


class XrSpatialProcessor:
    """
    Process spatial data with xarray DataArrays, including area calculations and geometry intersections.

    This class provides methods for:
    - Calculating pixel intersection proportions with geometries
    - Computing grid cell areas inspecified unit (default = 'ha')
    - Handling coordinate transformations and projections
    - Masking and weighting by geometry intersections

    Attributes:
        base_da (xr.DataArray): Base dataset for processing
        geometry (Union[BaseGeometry, List[BaseGeometry], None]): Shapely geometry or list of geometries
            for processing. If provided during initialization:
            - Single geometry if dissolve=True
            - List of geometries if dissolve=False
            - None if no geometry provided

    Example Usage:
        ```python
        # Initialize with base dataset only
        processor = XrSpatialProcessor(base_dataset)

        # Initialize with geometry file
        processor = XrSpatialProcessor(
            base_dataset,
            geom_source="area.geojson",
            dissolve=True
        )

        # Calculate masked areas
        masked = processor.create_mask_da()
        weights = processor.create_mask_weight_da()
        areas = processor.create_mask_weight_area_da()
        ```

    Notes:
        - All area calculations return values in specified unit (default = 'ha')
        - Geometries must be in the same CRS as the base dataset
        - Area calculations assume planar coordinates
        - Geometry processing handles both single and multiple features
    """

    def __init__(
        self,
        base_da: xr.DataArray,
        geom_source: Optional[Union[str, dict]] = None,
        dissolve: bool = True,
        unit: Union[str, AreaUnit] = "ha"
    ):
        """
        Initialize SpatialProcessor with a base dataset and optional GeoJSON input.

        Args:
            base_da (xr.DataArray): Base dataset for processing.
            geom_source (Union[str, dict], optional): Input geometry as either:
                - Path to a GeoJSON file (str)
                - A dictionary representing a GeoJSON object.
            dissolve (bool): Whether to dissolve (union) all geometries into one (default: True).
            unit (Union[str, AreaUnit]): The desired area unit (default = "ha").

        Raises:
            ValueError: If the GeoJSON is invalid.

        Example:
            processor = XrSpatialProcessor(base_dataset, geom_source="area.geojson", dissolve=True)
        """
        self.da = base_da
        self.unit = Units.get_unit(unit)
        self.geom = None
        self.geom_bbox = None

        if geom_source is not None:
            # Load GeoJSON from a file or directly use provided dictionary
            if isinstance(geom_source, str):
                with open(geom_source, 'r') as f:
                    geojson_data = json.load(f)
            elif isinstance(geom_source, dict):
                geojson_data = geom_source
            else:
                raise ValueError(
                    "geom_source must be a file path or a GeoJSON dictionary")

            # Extract geometries using shapely
            geoms = []
            if 'features' in geojson_data:
                for feature in geojson_data['features']:
                    if 'geometry' in feature and feature['geometry'] is not None:
                        geoms.append(shape(feature['geometry']))
            else:
                geoms.append(shape(geojson_data))

            if not geoms:
                raise ValueError(
                    "No valid geometries found in the GeoJSON input.")

            if dissolve:
                # Union all geometries into one
                union_geom = geoms[0]
                for g in geoms[1:]:
                    union_geom = union_geom.union(g)
                self.geom = [union_geom]
            else:
                self.geom = geoms

            # Compute the bounding box from the (dissolved) geometry
            self.geom_bbox = self.geom[0].bounds  # (minx, miny, maxx, maxy)

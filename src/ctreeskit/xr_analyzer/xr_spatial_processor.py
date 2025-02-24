from typing import Union, Optional
import geopandas as gpd
from pyproj import Proj
import xarray as xr
import numpy as np
from shapely.geometry import box
from rasterio.enums import Resampling, MergeAlg
import rioxarray as rio
from rasterio import features
from affine import Affine
from .common import Units, AreaUnit


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
        geom_source: Optional[Union[str, gpd.GeoDataFrame]] = None,
        dissolve: bool = True,
        unit: Union[str, AreaUnit] = "ha",
        generate_masks=True
    ):
        """
        Initialize SpatialProcessor with a base dataset and optional geometry.

        Args:
            base_da (xr.DataArray): Base dataset for processing
            geom_source (Union[str, gpd.GeoDataFrame], optional): Input geometry as either:
                - Path to GeoJSON/GPKG file (str)
                - GeoDataFrame (gpd.GeoDataFrame)
            dissolve (bool): Whether to dissolve all geometries into one (default: True)

        Raises:
            ValueError: If CRS don't match or geometry is invalid

        Example:
            ```python
            # Initialize with just base dataset
            processor = XrSpatialProcessor(base_dataset)

            # Initialize with geometry file
            processor = XrSpatialProcessor("area.geojson", dissolve=True)

            # Initialize with GeoDataFrame
            processor = XrSpatialProcessor(gdf, dissolve=False)
            ```
        """
        self.da = base_da.squeeze()
        self.unit = Units.get_unit(unit)
        self.geom = None
        self.geom_bbox = None
        self.binary_mask = None  # Initialize mask attribute
        self.da_clipped = None  # Initialize clipped data attribute

        if geom_source is not None:
            # Handle different input types
            if isinstance(geom_source, str):
                gdf = gpd.read_file(geom_source)
            elif isinstance(geom_source, gpd.GeoDataFrame):
                gdf = geom_source.copy()
            else:
                raise ValueError(
                    "Geometry source must be a file path or GeoDataFrame")

            # Ensure GeoDataFrame has CRS
            if gdf.crs is None:
                raise ValueError("Input geometry has no CRS information")

            # Get base dataset CRS
            base_crs = self.da.rio.crs
            if base_crs is None:
                raise ValueError("Base dataset has no CRS information")

            # Reproject geometry if needed
            if gdf.crs != base_crs:
                gdf = gdf.to_crs(base_crs)

            self.geom_bbox = gdf.total_bounds
            # Store geometry based on dissolve parameter
            if dissolve:
                geometry = gdf.geometry.union_all()
                self.geom = [geometry]  # Always store as list
            else:
                self.geom = gdf.geometry.tolist()  # Already a list
            if generate_masks:
                # Pre-compute binary mask using optimized rasterization
                self.binary_mask = self._create_fast_binary_mask()

                # Pre-compute clipped data using the binary mask
                self.da_clipped = self._create_fast_clipped_data()

    def reproject_match_target_da(self, target_da: xr.DataArray) -> xr.DataArray:
        """
        Reproject and resample source DataArray to match target DataArray's resolution and extent.

        Args:
            target_da (xr.DataArray): Target data whose grid to match

        Returns:
            xr.DataArray: Reprojected data matching target grid

        Example:
            >>> reprojected = reproject_match(agb_data, deforestation_data)
        """
        # Reproject to match target
        reprojected = self.da.rio.reproject_match(
            target_da,
            resampling=Resampling.nearest
        )

        return reprojected

    def _create_fast_binary_mask(self) -> xr.DataArray:
        """Internal method for fast binary mask creation clipped to geometry bounds"""
        if self.geom_bbox is None:
            raise ValueError("No geometry bounds available")

        # Get bbox coordinates
        minx, miny, maxx, maxy = self.geom_bbox

        # Slice the DataArray to bbox extent first
        da_subset = self.da.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny)  # Note: y is typically in descending order
        )

        # Get dimensions and transform for the subset
        height = da_subset.rio.height
        width = da_subset.rio.width
        transform = da_subset.rio.transform()

        # Fast rasterization with minimal parameters
        rasterized = features.rasterize(
            [(geom, 1) for geom in self.geom],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
            merge_alg=MergeAlg.replace
        )

        return xr.DataArray(
            rasterized,
            dims=('y', 'x'),
            coords={
                'y': da_subset.y,
                'x': da_subset.x
            }
        )

    def _create_fast_clipped_data(self) -> xr.DataArray:
        """Internal method for fast data clipping"""
        # Direct masking without interpolation when possible
        # Get bbox coordinates
        minx, miny, maxx, maxy = self.geom_bbox

        da_subset = self.da.sel(
            x=slice(minx, maxx),
            y=slice(maxy, miny))

        return da_subset.where(self.binary_mask == 1, 0)

    def create_binary_geom_mask_da(
        self,
        fit_to_geometry: bool = True
    ) -> xr.DataArray:
        """
        Rasterize stored geometries using base dataset's transform and dimensions.

        Args:
            fit_to_geometry (bool, optional): If True, output extent will match geometry bounds.
                If False, uses full extent of base dataset.

        Returns:
            xr.DataArray: Rasterized version of the geometry where:
                - Pixels intersecting geometry have value 1
                - Pixels outside geometry have value 0
                - Resolution matches base dataset
                - CRS matches base dataset

        Raises:
            ValueError: If self.geom is None
        """
        if self.geom is None:
            raise ValueError(
                "No geometry available. Initialize with geometry or use process_geometry first.")
        # Use pre-computed mask if available and not fitting to geometry
        if fit_to_geometry and self.binary_mask is not None:
            return self.binary_mask

        if fit_to_geometry and self.binary_mask is None:
            self.binary_mask = self._create_fast_binary_mask()
            return self.binary_mask

        # Otherwise create new mask
        height = self.da.rio.height
        width = self.da.rio.width
        transform = self.da.rio.transform()

        rasterized = features.rasterize(
            [(geom, 1) for geom in self.geom],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
            merge_alg=MergeAlg.replace
        )

        coords = {
            'y': self.da.y,
            'x': self.da.x
        }

        # Create DataArray with appropriate coordinates
        result = xr.DataArray(
            rasterized,
            dims=('y', 'x'),
            coords=coords,
            attrs={'_FillValue': 0}
        )

        # Copy CRS from base dataset
        result.rio.write_crs(self.da.rio.crs, inplace=True)

        return result
        # Create coordinates
        # if fit_to_geometry:
        #     y_coords = np.arange(height) * (-resolution) + bounds[3]
        #     x_coords = np.arange(width) * resolution + bounds[0]
        # else:
        #     y_coords = self.da.y
        #     x_coords = self.da.x
        #        # Create DataArray
        # result = xr.DataArray(
        #     rasterized,
        #     dims=('y', 'x'),
        #     coords={
        #         'y': y_coords,
        #         'x': x_coords
        #     },
        #     attrs={'_FillValue': 0}
        # )
        # # # Copy CRS
        # result.rio.write_crs(self.da.rio.crs, inplace=True)

        # return result
    def create_weighted_geom_mask_da_v2(self) -> xr.DataArray:
        """Precise calculation of geometry overlap proportions"""
        if self.geom is None:
            raise ValueError("No geometry available")

        # Get binary mask first for efficiency
        binary_mask = self.create_binary_geom_mask_da()
        transform = binary_mask.rio.transform()

        # Vectorized approach for initial mask
        shapes = [(geom, 1) for geom in self.geom]
        height, width = binary_mask.shape

        # Use higher precision float64 for intersection calculations
        weights = np.zeros((height, width), dtype=np.float64)
        pixel_area = abs(transform.a * transform.e)

        # Only process pixels that intersect (optimization)
        y_idx, x_idx = np.nonzero(binary_mask.data)
        for y, x in zip(y_idx, x_idx):
            x_min, y_min = transform * (x, y)
            x_max, y_max = transform * (x + 1, y + 1)
            pixel_geom = box(x_min, y_min, x_max, y_max)

            # Calculate precise intersection
            total_intersection = sum(geom.intersection(
                pixel_geom).area for geom in self.geom)
            weights[y, x] = min(total_intersection / pixel_area, 1.0)

        return xr.DataArray(
            weights,
            coords=binary_mask.coords,
            dims=('y', 'x'),
            attrs={'units': 'proportion'}
        )

    def create_weighted_geom_mask_da(self) -> xr.DataArray:
        """
        Calculate the proportion of each pixel that intersects with the geometry.
        Uses self.geometry which can be either a single geometry or list of geometries.

        Returns:
            xr.DataArray: Array containing pixel intersection proportions (0-1)
                where 1 means fully within geometry and 0 means no intersection

        Raises:
            ValueError: If self.geometry is None
        """
        if self.geom is None:
            raise ValueError(
                "No geometry available. Initialize with geometry or use process_geometry first.")
        # First clip to bounding box for efficiency
        masked = self.create_binary_geom_mask_da()

        # Get the pixel size and transform
        transform = masked.rio.transform()
        # Use transform coefficients directly
        pixel_size = abs(transform.a * transform.e)

        # Create an empty array for percentages
        percentage_array = np.zeros(masked.shape, dtype=np.float32)

        # Calculate percentages only for masked pixels
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                if masked.data[i, j] == 1:
                    # Create bounds using transform properly
                    x_min, y_min = transform * (j, i)
                    x_max, y_max = transform * (j + 1, i + 1)
                    pixel_geom = box(x_min, y_min, x_max, y_max)

                    # Calculate total intersection
                    total_intersection = sum(
                        geom.intersection(pixel_geom).area
                        for geom in self.geom
                    )
                    percentage_array[i, j] = min(
                        total_intersection / pixel_size, 1.0)

        # Convert to DataArray with same coordinates as masked
        return xr.DataArray(
            percentage_array,
            coords=masked.coords,
            dims=('y', 'x'),
            attrs={
                'units': 'proportion',
                'description': 'Pixel intersection proportions (0-1)'
            }
        )

    def create_area_mask_da(self, input_projection: int = 4236) -> xr.DataArray:
        """
        Calculate grid cell areas in specified unit (default = 'ha') given an input_projection

        Returns:
            xr.DataArray: Grid cell areas in specified unit (default = 'ha') with same coordinates as base dataset

        Raises:
            ValueError: If input_projection is not set
        """

        # Get coordinate values
        lat_center = self.da.y.values
        lon_center = self.da.x.values

        # Calculate differences
        diff_x = np.diff(lon_center)
        diff_y = np.diff(lat_center)

        # Calculate bounds
        x_bounds = np.concatenate([
            lon_center[0][None] - diff_x[0] / 2,
            lon_center[:-1] + diff_x / 2,
            lon_center[-1][None] + diff_x[-1] / 2,
        ])
        y_bounds = np.concatenate([
            lat_center[0][None] - diff_y[0] / 2,
            lat_center[:-1] + diff_y / 2,
            lat_center[-1][None] + diff_y[-1] / 2,
        ])

        # Create meshgrid
        xv, yv = np.meshgrid(x_bounds, y_bounds, indexing="xy")

        # Project coordinates
        p = Proj(f"EPSG:{input_projection}", preserve_units=False)
        x2, y2 = p(longitude=xv, latitude=yv)

        # Calculate areas
        diff_x2 = np.diff(x2, axis=1)
        diff_y2 = np.diff(y2, axis=0)
        grid_area = np.abs(diff_x2[:-1, :] * diff_y2[:, :-1])

        grid_area = Units.convert(grid_area, Units.M2, self.unit)

        return xr.DataArray(
            grid_area,
            coords={'y': self.da.y, 'x': self.da.x},
            dims=['y', 'x'],
            attrs={
                'units': self.unit.symbol,
                'description': f'Grid cell area in {self.unit.name}'
            }
        )

    def create_weighted_area_geom_mask_da(self, input_projection: int = 4236) -> xr.DataArray:
        """
        Calculate area of each pixel weighted by its intersection with a geometry.

        Combines geometry intersection weights with grid cell areas to calculate
        the effective area of each pixel that intersects with the input geometry.

        Args:
            geom (dict): GeoJSON-like geometry dictionary
            input_projection (int, optional): EPSG code for area calculations, defaults to 4236

        Returns:
            xr.DataArray: Array containing weighted areas in specified unit (default = 'ha'), where:
                - Values outside geometry are 0
                - Values inside geometry are area * intersection percentage
                - Coordinates and dimensions match base dataset

        Example:
            For a 1ha pixel that is 75% covered by the geometry:
            weighted_area = 1ha * 0.75 = 0.75ha
        """
        # Get geometry weights (0-1)
        weights = self.create_weighted_geom_mask_da()

        # Calculate areas
        areas = self.create_area_mask_da(input_projection)

        # Use masked operations for better precision
        mask = weights > 0
        weighted_areas = xr.where(
            mask,
            # Use higher precision for multiplication
            areas * weights.astype(np.float64),
            0
        )
        # Update attributes
        weighted_areas.attrs.update({
            'units': self.unit.symbol,
            'description': f'Geometry-weighted grid cell area in {self.unit.name}'
        })

        return weighted_areas

    def create_clipped_da_vector(self) -> xr.DataArray:
        """
        Mask base dataset using self.geometry.

        Uses self.geometry which can be either a single geometry or list of geometries.
        All geometries should already be in the correct CRS from initialization.

        Returns:
            xr.DataArray: Masked array where pixels outside geometry are masked

        Raises:
            ValueError: If self.geometry is None
        """
        if self.geom is None:
            raise ValueError(
                "No geometry available. Initialize with geometry or use process_geometry first.")

        # First clip to bounding box for efficiency
        bbox_clipped = self.da.rio.clip_box(
            minx=self.geom_bbox[0],
            miny=self.geom_bbox[1],
            maxx=self.geom_bbox[2],
            maxy=self.geom_bbox[3]
        )

        # Mask the dataset with all geometries
        masked = bbox_clipped.rio.clip(
            self.geom,
            all_touched=True,
            from_disk=True
        )

        return masked

    def create_clipped_da_raster(self) -> xr.DataArray:
        """
        Mask base dataset using a binary mask.

        Args:
            binary_mask (xr.DataArray): Binary mask where 1 indicates pixels to keep

        Returns:
            xr.DataArray: Masked array where pixels outside geometry are masked

        Raises:
            ValueError: If self.geometry is None
        """
        if self.geom is None:
            raise ValueError(
                "No geometry available. Initialize with geometry or use process_geometry first.")

        # Mask the dataset with all geometries
        binary_mask = self.create_binary_geom_mask_da()
        # Slice base dataset to match mask coordinates
        sliced = self.da.interp(
            x=binary_mask.x,
            y=binary_mask.y,
            method='nearest',
            kwargs={'fill_value': 0}
        )
        # Apply mask
        masked = sliced.where(binary_mask == 1, 0)

        return masked

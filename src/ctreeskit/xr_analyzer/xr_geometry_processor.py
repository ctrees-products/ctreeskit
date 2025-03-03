from typing import Union, Optional
import geopandas as gpd
import xarray as xr
import numpy as np
from shapely.geometry import box
from rasterio import features
from rasterio.enums import MergeAlg
import rioxarray as rio
from affine import Affine
from .common import MaskType, Units
import pyproj
from shapely.ops import transform, unary_union
import dask
import dask.array as da
from rasterio.mask import mask
from pyproj import Proj

import warnings


class XrGeometryProcessor:
    """
    Processes a given geometry (or geometries) and provides functions that, given a raster,
    generate masks and weighted area calculations.

    The geometry is passed in during initialization. Raster-specific operations are provided
    as methods that take a DataArray as input.
    """

    def __init__(
        self,
        geom_source: Union[str, gpd.GeoDataFrame],
        dissolve: bool = True
    ):
        geom, crs, bbox = self._initialize_geometry(geom_source, dissolve)
        self.geom_crs = crs
        self.geom_bbox = bbox
        self.geom = geom
        self.geom_area = self._calculate_geometry_area()
        # Cache for different raster resolutions
        # Will store (transform, shape) -> (mask, mask_type)
        self._mask_cache = {}

    # Cache management
    def _get_raster_key(self, raster: xr.DataArray) -> tuple:
        """Generate a cache key based on raster properties."""
        raster_transform = raster.rio.transform()
        return (
            raster_transform.a,  # x resolution
            raster_transform.e,  # y resolution
            raster_transform.c,  # x origin
            raster_transform.f,  # y origin
            raster.shape  # grid dimensions
        )

    def _get_cached_mask(self, raster: xr.DataArray) -> Optional[tuple]:
        """Get cached mask for given raster properties."""
        key = self._get_raster_key(raster)
        return self._mask_cache.get(key)

    def _cache_mask(self, raster: xr.DataArray, mask: xr.DataArray, mask_type: MaskType):
        """Cache mask for given raster properties."""
        key = self._get_raster_key(raster)
        self._mask_cache[key] = (mask, mask_type)

    # Core geometry operations

    def _initialize_geometry(self, geom_source, dissolve):
        """Load and validate geometry source."""
        if isinstance(geom_source, str):
            gdf = gpd.read_file(geom_source)
        elif isinstance(geom_source, gpd.GeoDataFrame):
            gdf = geom_source.copy()
        else:
            raise ValueError(
                "Geometry source must be a file path or GeoDataFrame")

        if gdf.crs is None:
            raise ValueError("Input geometry has no CRS information")
        # Store geometry as a list for consistency.
        if dissolve:
            self.geom = [gdf.geometry.unary_union]
        else:
            self.geom = gdf.geometry.tolist()
        return self.geom, gdf.crs, gdf.total_bounds

    def _calculate_geometry_area(self, target_epsg: int = 3857) -> float:
        """
        Compute the area of the stored geometry in square meters.
        The geometry is reprojected from its original CRS (stored in self.geom_crs)
        to the target EPSG (default EPSG:3857) before calculating area.
        """
        # Get the union of the geometries if more than one exists.
        if len(self.geom) > 1:
            union_geom = unary_union(self.geom)
        else:
            union_geom = self.geom[0]

        # Set up coordinate transformation from the source CRS to target CRS (default EPSG:3857)
        target_crs = pyproj.CRS.from_epsg(target_epsg)
        transformer = pyproj.Transformer.from_crs(
            self.geom_crs, target_crs, always_xy=True).transform

        # Project the geometry into the target CRS
        projected_geom = transform(transformer, union_geom)
        self.geom_area = projected_geom.area
        # Return the area in square meters.
        return projected_geom.area

    # Spatial subsetting and alignment
    def _extract_spatial_parameters(self, spatial_raster: xr.DataArray) -> (np.ndarray, np.ndarray, tuple, Affine, xr.DataArray):
        """
        Extract spatial parameters from input raster for geometry processing.

        Parameters
        ----------
        spatial_raster : xr.DataArray
            Input raster from which to extract spatial parameters

        Returns
        -------
        tuple
            Contains:
            - new_y : np.ndarray
                Y-coordinates (latitude) of the raster pixels
            - new_x : np.ndarray
                X-coordinates (longitude) of the raster pixels
            - shape : tuple
                (height, width) dimensions of the raster
            - transform : Affine
                Affine transformation matrix for the raster
            - spatial_raster : xr.DataArray
                Input raster (possibly subset)
        """
        height, width = spatial_raster.shape
        raster_transform = spatial_raster.rio.transform()
        # Compute coordinates from the subset itself (or you can compute centers)
        new_y = spatial_raster.y.values[:height]
        new_x = spatial_raster.x.values[:width]
        return new_y, new_x, (height, width), raster_transform, spatial_raster

    def _calculate_pixel_centers(self, raster_transform: Affine, shape: tuple) -> (np.ndarray, np.ndarray):
        """
        Calculate center coordinates for each pixel in the raster grid.

        Parameters
        ----------
        raster_transform : Affine
            Affine transformation matrix defining the raster's georeference
        shape : tuple
            (height, width) dimensions of the raster

        Returns
        -------
        tuple
            Contains:
            - new_y : np.ndarray
                Y-coordinates of pixel centers
            - new_x : np.ndarray
                X-coordinates of pixel centers

        Notes
        -----
        Uses the affine transformation to convert pixel indices to real-world coordinates,
        adding 0.5 to get center points rather than corners.
        """
        height, width = shape
        new_x = raster_transform.c + \
            raster_transform.a * (np.arange(width) + 0.5)
        new_y = raster_transform.f + \
            raster_transform.e * (np.arange(height) + 0.5)
        return new_y, new_x

    def _align_coordinates(self, da: xr.DataArray, spatial_raster: xr.DataArray) -> xr.DataArray:
        """
        Ensure coordinate systems match between two DataArrays.

        Parameters
        ----------
        da : xr.DataArray
            DataArray to be aligned
        spatial_raster : xr.DataArray
            Reference DataArray with desired coordinate system

        Returns
        -------
        xr.DataArray
            Aligned DataArray with coordinates matching spatial_raster

        Notes
        -----
        Performs both CRS alignment (reprojection if needed) and
        coordinate assignment to ensure exact spatial matching.
        """
        # If CRS or transform differ, reproject:
        if da.rio.crs != spatial_raster.rio.crs or da.rio.transform() != spatial_raster.rio.transform():
            da = da.rio.reproject_match(spatial_raster)
        # Reassign coordinates from spatial_raster:
        return da.assign_coords({'y': spatial_raster.y, 'x': spatial_raster.x})

    def subset_to_bbox(self, raster: xr.DataArray, drop_time=False) -> xr.DataArray:
        """
        Subset input raster to the geometry's bounding box.

        Parameters
        ----------
        raster : xr.DataArray
            Input raster to be subset
        drop_time : bool, default=False
            If True, removes time dimension from result

        Returns
        -------
        xr.DataArray
            Spatially subset raster matching geometry's bounding box

        Notes
        -----
        Uses rioxarray's clip_box for efficient spatial subsetting.
        Maintains CRS and spatial properties of the input raster.
        If time dimension exists and drop_time=True, returns first time slice.
        """
        # Get the bounding box coordinates
        minx, miny, maxx, maxy = self.geom_bbox

        # First clip to the geometry's bounding box
        clipped = raster.rio.clip_box(
            minx=minx, miny=miny, maxx=maxx, maxy=maxy)
        if drop_time and 'time' in clipped.dims:
            return clipped.isel(time=0)

        return clipped

    # Mask generation

    def clip_raster_to_geom(self, raster: xr.DataArray, binary: bool = True) -> xr.DataArray:
        """
        Clip the provided raster using the stored geometry.

        Parameters
        ----------
        raster : xr.DataArray
            The input raster to clip
        binary : bool, default=True
            If True, uses binary (0/1) masking
            If False, uses weighted masking based on intersection proportions

        Returns
        -------
        xr.DataArray
            Clipped raster matching the geometry extent
            Values outside geometry are set to 0/NaN

        Notes
        -----
        Uses rioxarray's clip functionality with memory-efficient processing.
        Maintains CRS and spatial properties of the input raster.
        """
        # Prepare spatial subset if needed
        spatial_raster = self.subset_to_bbox(raster)

        # Convert geometries to GeoJSON format for clipping
        geoms = [g.__geo_interface__ for g in self.geom]

        # Clip the raster using rioxarray
        result = spatial_raster.rio.clip(
            geoms,
            crs=self.geom_crs,
            all_touched=True,
            drop=True,
            from_disk=True  # More memory efficient
        )

        # If using weighted mask, apply the weights from cache
        if not binary:
            mask_type = MaskType.WEIGHTED
            cached = self._get_cached_mask(raster)
            if cached and cached[1] == mask_type:
                weights = cached[0]
            else:
                weights = self.create_proportion_geom_mask(raster)

            # Ensure mask and result have same coordinates
            result = result.rio.reproject_match(weights)
            result = result * weights

        return result

    def create_binary_geom_mask(self, raster: xr.DataArray) -> xr.DataArray:
        """
        Create a binary (0/1) mask from the geometry.

        Parameters
        ----------
        raster : xr.DataArray
            Reference raster for output resolution and extent

        Returns
        -------
        xr.DataArray
            Binary mask where:
            1 = pixel intersects with geometry
            0 = pixel outside geometry

        Notes
        -----
        Uses rasterio's rasterize function with all_touched=True.
        Mask is aligned to input raster's coordinate system.
        Sets self.geom_mask and self.mask_type to BINARY.
        """
        if self.geom is None:
            raise ValueError("No geometry available.")

        cached = self._get_cached_mask(raster)
        if cached and cached[1] == MaskType.BINARY:
            return cached[0]
        # Work on the spatial subset (drop time if present)
        spatial_raster = self.subset_to_bbox(raster, True)

        # Get clipping parameters and the spatial subset corresponding to the bbox.
        _, _, out_shape, raster_transform, subset = self._extract_spatial_parameters(
            spatial_raster)

        # Rasterize the geometry using the determined shape and transform.
        rasterized = features.rasterize(
            [(geom, 1) for geom in self.geom],
            out_shape=out_shape,
            transform=raster_transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
            merge_alg=MergeAlg.replace
        )

        # Compute new coordinates from the transform. (Alternatively, you could use the subset’s coordinates.)
        # Here we choose to compute pixel centers.
        comp_y, comp_x = self._calculate_pixel_centers(
            raster_transform, out_shape)
        coords = {'y': comp_y, 'x': comp_x}

        result = xr.DataArray(
            rasterized,
            dims=('y', 'x'),
            coords=coords,
            attrs={'_FillValue': 0}
        )
        result.rio.write_crs(subset.rio.crs, inplace=True)

        # Use reproject_match with the subset so that the output
        # now has the bbox extent rather than that of the full raster.
        aligned_mask = result.rio.reproject_match(subset)

        self._cache_mask(raster, aligned_mask, MaskType.BINARY)

        return aligned_mask

    def create_proportion_geom_mask(self, raster: xr.DataArray, pixel_ratio=.001, overwrite=False) -> xr.DataArray:
        """
        Create a weighted mask based on geometry intersection proportions.

        Parameters
        ----------
        raster : xr.DataArray
            Reference raster for output resolution and extent
        pixel_ratio : float, default=0.001
            Minimum ratio of pixel area to geometry area (0.1% default)
        overwrite : bool, default=False
            If True, bypasses the pixel_ratio check

        Returns
        -------
        xr.DataArray
            Weighted mask where values 0-1 represent:
            1.0 = pixel fully within geometry
            0.0 = pixel outside geometry
            0.0-1.0 = proportion of pixel intersecting geometry

        Notes
        -----
        Falls back to binary mask if pixel_ratio check fails.
        Uses exact geometry intersection calculations.
        Sets self.geom_mask and self.mask_type to WEIGHTED.
        """
        # Use existing caching and binary mask creation
        cached = self._get_cached_mask(raster)
        if cached and cached[1] == MaskType.WEIGHTED and not overwrite:
            return cached[0]

        binary_mask = self.create_binary_geom_mask(raster)

        raster_transform = binary_mask.rio.transform()
        pixel_size = abs(raster_transform.a * raster_transform.e)
        percentage_array = np.zeros(binary_mask.shape, dtype=np.float32)
        # When overwrite is False, enforce the pixel_ratio check.
        if not overwrite:
            ratio = pixel_size / self.geom_area
            if ratio < pixel_ratio:
                warnings.warn(
                    f"(pixel area ratio {ratio:.3e} is below {pixel_ratio*100:.3e}% of the project area). "
                    "Weighted mask computation skipped; binary mask automatically set to self.geom_mask."
                    "Use overwrite=True to utilize porportion-based mask computation.",
                    UserWarning
                )
                return binary_mask

        # Loop over nonzero pixels:
        y_idx, x_idx = np.nonzero(binary_mask.data)
        for y, x in zip(y_idx, x_idx):
            x_min, y_min = raster_transform * (x, y)
            x_max, y_max = raster_transform * (x + 1, y + 1)
            pixel_geom = box(x_min, y_min, x_max, y_max)
            total_int = sum(geom.intersection(
                pixel_geom).area for geom in self.geom)
            percentage_array[y, x] = min(total_int / pixel_size, 1.0)

        result = xr.DataArray(
            percentage_array,
            coords=binary_mask.coords,
            dims=binary_mask.dims,
            attrs={'units': 'proportion',
                   'description': 'Pixel intersection proportions (0-1)'}
        )
        self._cache_mask(raster, result, MaskType.WEIGHTED)
        return result

    # Area calculations

    def create_area_geom_mask(self, raster: xr.DataArray, binary: bool = True, unit=Units.M2, input_projection: int = 4236) -> xr.DataArray:
        """
        Calculate areas for each raster cell within the geometry.

        Parameters
        ----------
        raster : xr.DataArray
            Input raster defining the grid
        binary : bool, default=True
            If True, uses binary masking
            If False, uses weighted intersection areas
        unit : Units, default=Units.M2
            Output area units (e.g., m², ha, km²)
        input_projection : int, default=4236
            EPSG code for area calculations (default WGS84)

        Returns
        -------
        xr.DataArray
            Grid cell areas in specified units
            Values are 0 outside geometry
            Inside geometry:
                binary=True: full cell area
                binary=False: proportional area

        Notes
        -----
        Combines geometry mask with calculated grid cell areas.
        Handles coordinate systems and unit conversions.
        Sets self.geom_mask and self.mask_type to AREA.
        """
        # Get weights and area from raster (note: area calculation is handled in the raster class below)
        # circular dependency; adjust if needed
        mask_type = MaskType.BINARY if binary else MaskType.WEIGHTED
        unit = Units.get_unit(unit)
        cached = self._get_cached_mask(raster)
        if cached and cached[1] == mask_type:
            base_mask = cached[0]
        else:
            base_mask = (self.create_binary_geom_mask(raster) if binary
                         else self.create_proportion_geom_mask(raster))
        weights = base_mask
        areas = self.create_pixel_areas(
            raster, input_projection=input_projection)
        mask = weights > 0
        weighted_areas = xr.where(mask, areas * weights.astype(np.float64), 0)
        weighted_areas.attrs.update({
            'units': unit.symbol,
            'description': f'Geometry-weighted grid cell area in {unit.name}'
        })
        weighted_areas.rio.write_crs(raster.rio.crs, inplace=True)
        self._cache_mask(raster, weighted_areas, MaskType.AREA)

        return weighted_areas

    def create_proportion_geom_mask_dask(self, raster: xr.DataArray, pixel_ratio=.001, overwrite=False) -> xr.DataArray:
        """
        Create a weighted mask based on geometry intersection proportions using Dask.
        Parallel version of create_proportion_geom_mask.

        Parameters
        ----------
        raster : xr.DataArray
            Reference raster for output resolution and extent
        pixel_ratio : float, default=0.001
            Minimum ratio of pixel area to geometry area (0.1% default)
        overwrite : bool, default=False
            If True, bypasses the pixel_ratio check

        Returns
        -------
        xr.DataArray
            Weighted mask where values 0-1 represent:
            1.0 = pixel fully within geometry
            0.0 = pixel outside geometry
            0.0-1.0 = proportion of pixel intersecting geometry

        Notes
        -----
        - Uses Dask for parallel processing
        - Falls back to binary mask if pixel_ratio check fails
        - Reuses binary mask and caching from other methods
        """
        # Use existing caching and binary mask creation
        cached = self._get_cached_mask(raster)
        if cached and cached[1] == MaskType.WEIGHTED and not overwrite:
            return cached[0]

        binary_mask = self.create_binary_geom_mask(raster)
        raster_transform = binary_mask.rio.transform()
        pixel_size = abs(raster_transform.a * raster_transform.e)
        shape = binary_mask.shape

        # Reuse pixel ratio check logic
        if not overwrite:
            ratio = pixel_size / self.geom_area
            if ratio < pixel_ratio:
                warnings.warn(
                    f"(pixel area ratio {ratio:.3e} is below {pixel_ratio*100:.3e}% of the project area). "
                    "Weighted mask computation skipped; binary mask automatically set to self.geom_mask."
                    "Use overwrite=True to utilize porportion-based mask computation.",
                    UserWarning
                )
                return binary_mask

        @dask.delayed
        def compute_pixel_weight(y: int, x: int) -> float:
            """Compute intersection proportion for a single pixel."""
            x_min, y_min = raster_transform * (x, y)
            x_max, y_max = raster_transform * (x + 1, y + 1)
            pixel_geom = box(x_min, y_min, x_max, y_max)
            total_int = sum(geom.intersection(
                pixel_geom).area for geom in self.geom)
            return min(total_int / pixel_size, 1.0)

        # Create dask array from binary mask
        y_idx, x_idx = np.nonzero(binary_mask.data)
        delayed_values = np.zeros(shape)

        # Only compute weights for pixels that intersect geometry
        for y, x in zip(y_idx, x_idx):
            delayed_values[y, x] = compute_pixel_weight(y, x)

        # Convert to dask array
        dask_array = da.from_array(delayed_values, chunks='auto')

        # Create xarray DataArray with same coordinates as binary mask
        result = xr.DataArray(
            dask_array,
            coords=binary_mask.coords,
            dims=binary_mask.dims,
            attrs={'units': 'proportion',
                   'description': 'Pixel intersection proportions (0-1)'}
        )

        # Reuse coordinate alignment
        weighted_mask = self._align_coordinates(result, binary_mask)

        # Cache the result
        self._cache_mask(raster, weighted_mask, MaskType.WEIGHTED)

        return weighted_mask

    def create_pixel_areas(self, raster, unit=Units.M2, input_projection: int = 4236) -> xr.DataArray:
        """
        Calculate the area of each grid cell in the raster.

        Parameters
        ----------
        raster : xr.DataArray
            Input raster defining the grid
        unit : Units, default=Units.M2
            Output area units (e.g., m², ha, km²)
        input_projection : int, default=4236
            EPSG code for area calculations (default WGS84)

        Returns
        -------
        xr.DataArray
            Grid cell areas in specified units
            Accounts for Earth's curvature
            Areas vary with latitude

        Notes
        -----
        Uses coordinate bounds to compute exact areas.
        Projects coordinates for accurate area calculation.
        Independent of geometry (full raster extent).
        """
        spatial_raster = self.subset_to_bbox(raster)
        lat_center = spatial_raster.y.values
        lon_center = spatial_raster.x.values

        diff_x = np.diff(lon_center)
        diff_y = np.diff(lat_center)
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
        xv, yv = np.meshgrid(x_bounds, y_bounds, indexing="xy")
        p = Proj(f"EPSG:{input_projection}", preserve_units=False)
        x2, y2 = p(longitude=xv, latitude=yv)
        diff_x2 = np.diff(x2, axis=1)
        diff_y2 = np.diff(y2, axis=0)
        grid_area = np.abs(diff_x2[:-1, :] * diff_y2[:, :-1])
        if unit:
            grid_area = Units.convert(grid_area, Units.M2, unit)
        return xr.DataArray(
            grid_area,
            coords={'y': spatial_raster.y, 'x': spatial_raster.x},
            dims=['y', 'x'],
            attrs={
                'units': unit.symbol,
                'description': f'Grid cell area in {unit.name}'
            }
        )

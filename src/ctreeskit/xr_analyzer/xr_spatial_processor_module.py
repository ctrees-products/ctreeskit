import json
import warnings
from typing import Union, Optional, List, Protocol

import numpy as np
import xarray as xr
import pandas as pd
import s3fs
import pyproj
from pyproj import Proj, Geod
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from rasterio.enums import Resampling

# A simple protocol to type-check geometry-like objects.


class GeometryLike(Protocol):
    geom_type: str


class GeometryData:
    """
    Container for spatial geometry information.

    Attributes
    ----------
    geom : Optional[List[GeometryLike]]
         List of geometry objects (usually Shapely geometries).
    geom_crs : Optional[str]
         Coordinate reference system as an EPSG string (e.g., "EPSG:4326").
    geom_bbox : Optional[tuple]
         Bounding box of the geometry (minx, miny, maxx, maxy).
    geom_area : Optional[float]
         Area of the geometry (in m² or ha, depending on conversion).
    """
    geom: Optional[List[GeometryLike]]
    geom_crs: Optional[str]
    geom_bbox: Optional[tuple]
    geom_area: Optional[float]

    def __init__(self, geom: Optional[List[GeometryLike]] = None,
                 geom_crs: Optional[str] = None,
                 geom_bbox: Optional[tuple] = None,
                 geom_area: Optional[float] = None):
        self.geom = geom
        self.geom_crs = geom_crs
        self.geom_bbox = geom_bbox
        self.geom_area = geom_area


GeometrySource = Union[str, "GeometryLike", List["GeometryLike"]]
ExtendedGeometryInput = Union["GeometryData", GeometrySource,
                              List[Union["GeometryData", "GeometryLike"]]]


def process_geometry(geom_source: GeometrySource,
                     dissolve: bool = True, output_in_ha=True):
    """
    Load, validate, and process a geometry source into a standardized GeometryData object.

    The geom_source may be one of:
      - A file path (local or S3 URI) pointing to a GeoJSON file.
      - A single geometry (that implements the 'geom_type' attribute).
      - A list of geometries.

    If dissolve is True the geometries are merged into a single object and the bounding box is computed
    accordingly.

    Parameters
    ----------
    geom_source : str or GeometryLike or list of GeometryLike
         The input geometry source.
    dissolve : bool, default True
         If True, all geometries are dissolved into a single geometry.
    output_in_ha : bool, default True
         If True, converts the computed area from square meters to hectares.

    Returns
    -------
    GeometryData
         An object containing:
             - geom: a list of geometry(ies) (dissolved if requested)
             - geom_crs: the coordinate reference system (string)
             - geom_bbox: the bounding box (minx, miny, maxx, maxy)
             - geom_area: the computed area (converted if output_in_ha is True)
    """
    geometries = None
    crs = None

    if isinstance(geom_source, str):
        if geom_source.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(geom_source, 'r') as f:
                geojson_dict = json.load(f)
        else:
            with open(geom_source, 'r') as f:
                geojson_dict = json.load(f)
        geometries = [shape(feature['geometry'])
                      for feature in geojson_dict.get('features', [])]
        crs = geojson_dict.get('crs', {}).get(
            'properties', {}).get('name', None)
        if crs is None:
            raise ValueError("Input geometry has no CRS information")
    elif isinstance(geom_source, list) and all(hasattr(g, 'geom_type') for g in geom_source):
        geometries = geom_source
        crs = "EPSG:4326"  # default CRS
    elif hasattr(geom_source, 'geom_type'):
        geometries = [geom_source]
        crs = "EPSG:4326"
    else:
        raise ValueError(
            "Geometry source must be a file path, a list of geometries, or a geometry")

    if dissolve:
        union_geom = unary_union(geometries)
        geom = [union_geom]
        geom_bbox = union_geom.bounds
    else:
        geom = geometries
        geom_bbox = unary_union(geometries).bounds

    area = _calculate_geometry_area(geom, crs)
    conversion = 1e-4 if output_in_ha else 1.0
    return GeometryData(geom=geom, geom_crs=crs, geom_bbox=geom_bbox,
                        geom_area=area * conversion)


def _calculate_geometry_area(geom: List, geom_crs: str, target_epsg: int = 6933) -> float:
    """
    Compute the area of a set of geometries in square meters using a target projection.

    Parameters
    ----------
    geom : list
         List of Shapely geometry objects.
    geom_crs : str
         Source coordinate reference system (e.g., "EPSG:4326").
    target_epsg : int, default 6933
         EPSG code for the target projection used to compute area accurately.

    Returns
    -------
    float
         Computed area in square meters.
    """
    if len(geom) > 1:
        union_geom = unary_union(geom)
    else:
        union_geom = geom[0]
    target_crs = pyproj.CRS.from_epsg(target_epsg)
    transformer = pyproj.Transformer.from_crs(
        geom_crs, target_crs, always_xy=True).transform
    projected_geom = transform(transformer, union_geom)
    return projected_geom.area


def clip_ds_to_bbox(input_ds: Union[xr.DataArray, xr.Dataset], bbox: tuple, drop_time: bool = False) -> xr.DataArray:
    """
    Clip a raster (DataArray or Dataset) to a given bounding box.

    Parameters
    ----------
    input_ds : xr.DataArray or xr.Dataset
         The input raster with valid spatial metadata.
    bbox : tuple
         Bounding box as (minx, miny, maxx, maxy).
    drop_time : bool, default False
         If True and the raster has a 'time' dimension, only the first time slice is returned.

    Returns
    -------
    xr.DataArray
         The raster clipped to the specified bounding box.

    Raises
    ------
    ValueError
        If bbox is not a tuple of length 4 or contains non-numeric values.
    """
    # Validate bbox
    if not isinstance(bbox, tuple) or len(bbox) != 4:
        raise ValueError(
            "bbox must be a tuple of length 4 (minx, miny, maxx, maxy).")
    if not all(isinstance(coord, (int, float)) for coord in bbox):
        raise ValueError(
            "All elements of bbox must be numeric (int or float).")

    minx, miny, maxx, maxy = bbox
    clipped = input_ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    if drop_time and 'time' in clipped.dims:
        return clipped.isel(time=0)
    return clipped


def clip_ds_to_geom(input_ds: Union[xr.DataArray, xr.Dataset], geom_source: ExtendedGeometryInput, all_touch=False) -> xr.DataArray:
    """
    Clip a raster to the extent of the provided geometry.

    The input geometry may be provided as a GeometryData instance or as a raw geometry source
    (e.g., a file path, a single geometry, or a list). If not a GeometryData instance, the geometry
    is processed via process_geometry.

    Parameters
    ----------
    input_ds : xr.DataArray or xr.Dataset
         The input raster to be clipped. Must contain spatial metadata.
    geom_source : ExtendedGeometryInput
         Either a GeometryData instance, a single geometry (or list), or a GeoJSON file path.
         The geometry objects should have a 'geom_type' attribute if not already wrapped.
    all_touch : bool, default False
         If True, includes all pixels touched by the geometry boundaries.

    Returns
    -------
    xr.DataArray
         Raster clipped to the geometry’s spatial extent. Areas outside the geometry are set to 0 or NaN.
    """
    if not isinstance(geom_source, GeometryData):
        geom_source = process_geometry(geom_source, True)
    geom = geom_source.geom
    bbox = geom_source.geom_bbox
    crs = geom_source.geom_crs

    # Prepare spatial subset if needed
    spatial_raster = clip_ds_to_bbox(input_ds, bbox)

    # Convert geometries to GeoJSON format for clipping
    geoms = [g.__geo_interface__ for g in geom]

    # Clip the raster using rioxarray
    result = spatial_raster.rio.clip(
        geoms,
        crs=crs,
        all_touched=all_touch,
        drop=True,
        from_disk=True  # More memory efficient
    )

    return result


def _measure(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute the geodesic distance (in meters) between two points on the WGS84 ellipsoid.

    Parameters
    ----------
    lat1 : float
        Latitude of the first point in decimal degrees.
    lon1 : float
        Longitude of the first point in decimal degrees.
    lat2 : float
        Latitude of the second point in decimal degrees.
    lon2 : float
        Longitude of the second point in decimal degrees.

    Returns
    -------
    float
        The geodesic distance between the two points in meters.
    """
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance


def reproject_match_ds(template_raster: Union[xr.DataArray, xr.Dataset],
                       target_raster: Union[xr.DataArray, xr.Dataset],
                       resampling_method=Resampling.nearest,
                       return_area_grid: bool = True,
                       output_in_ha: bool = True):
    """
    Align and resample a target raster to match the spatial grid of a template raster.

    The target raster is first clipped to the extent of the template raster and then reprojected
    so that its grid (extent, resolution, and transform) exactly matches that of the template.
    Optionally, a grid of cell areas is computed on the aligned raster.

    Parameters
    ----------
    template_raster : xr.DataArray or xr.Dataset
         The reference raster defining the target grid.
    target_raster : xr.DataArray or xr.Dataset
         The raster to be aligned and resampled.
    resampling_method : str, optional
         The resampling algorithm to use (e.g., "nearest", "bilinear").
    return_area_grid : bool, default True
         If True, returns a DataArray with grid cell areas.
    output_in_ha : bool, default True
         If True, computed areas will be converted to hectares; otherwise, areas are in square meters.

    Returns
    -------
    tuple
         A tuple (aligned_target, area_target) where:
         - aligned_target is the resampled target raster.
         - area_target is the grid of cell areas (or None if return_area_grid is False).
    """
    clipped_target = clip_ds_to_bbox(
        target_raster, template_raster.rio.bounds(), drop_time=True)
    aligned_target = clipped_target.rio.reproject_match(
        template_raster, resampling=resampling_method)
    area_target = None
    if return_area_grid:
        area_target = create_area_ds_from_degrees_ds(
            aligned_target, output_in_ha=output_in_ha)
    return aligned_target, area_target


def create_proportion_geom_mask(input_ds: xr.DataArray, geom_source: ExtendedGeometryInput, pixel_ratio=.001, overwrite=False) -> xr.DataArray:
    """
    Create a weighted mask for a raster based on the intersection proportions of its pixels with a geometry.

    This function calculates, for each pixel in the input raster, the proportion of the pixel area intersecting the
    provided geometry. If the pixel area (derived from the raster transform) is below a given ratio threshold
    (unless overwrite is True), the function returns a binary mask instead.

    Parameters
    ----------
    input_ds : xr.DataArray
         The input raster whose pixel intersection proportions are to be computed.
    geom_source : ExtendedGeometryInput
         Either a GeometryData instance or a raw geometry source (e.g., a file path, a single geometry, or a list).
         If not a GeometryData instance, it is processed via process_geometry.s
    pixel_ratio : float, default 0.001
         The minimum ratio of pixel area to geometry area required before performing a weighted computation;
         otherwise, a binary mask is returned.
    overwrite : bool, default False
         If True, bypasses the pixel_ratio check and always computes weighted proportions.

    Returns
    -------
    xr.DataArray
         A DataArray mask where each pixel value (between 0 and 1) represents the fraction of that pixel's area
         that intersects the geometry. Pixels with no intersection return 0.
    """
    # Use existing caching and binary mask creation
    if not isinstance(geom_source, GeometryData):
        geom_source = process_geometry(geom_source, True)
    geom = geom_source.geom

    raster_transform = input_ds.rio.transform()
    pixel_size = abs(raster_transform.a * raster_transform.e)
    percentage_array = np.zeros(input_ds.shape, dtype=np.float32)
    # When overwrite is False, enforce the pixel_ratio check.
    if not overwrite:
        ratio = pixel_size / geom.geom_area
        if ratio < pixel_ratio:
            warnings.warn(
                f"(pixel area ratio {ratio:.3e} is below {pixel_ratio*100:.3e}% of the project area). "
                "Weighted mask computation skipped; binary mask automatically set to self.geom_mask."
                "Use overwrite=True to utilize porportion-based mask computation.",
                UserWarning
            )
            clipped_raster = clip_ds_to_geom(geom_source, input_ds)
            return xr.where(clipped_raster.notnull(), 1, 0)

    clipped_raster = clip_ds_to_geom(geom_source, input_ds, all_touch=True)
    # Loop over nonzero pixels:
    y_idx, x_idx = np.nonzero(clipped_raster.data)
    for y, x in zip(y_idx, x_idx):
        x_min, y_min = raster_transform * (x, y)
        x_max, y_max = raster_transform * (x + 1, y + 1)
        pixel_geom = box(x_min, y_min, x_max, y_max)
        total_int = sum(geom.intersection(
            pixel_geom).area for geom in geom_source.geom)
        percentage_array[y, x] = min(total_int / pixel_size, 1.0)

    result = xr.DataArray(
        percentage_array,
        coords=clipped_raster.coords,
        dims=clipped_raster.dims,
        attrs={'units': 'proportion',
               'description': 'Pixel intersection proportions (0-1)'}
    )
    return result


def create_area_ds_from_degrees_ds(input_ds:  Union[xr.DataArray, xr.Dataset],
                                   high_accuracy: Optional[bool] = None,
                                   output_in_ha: bool = True) -> xr.DataArray:
    """
    Calculate cell areas for a geographic raster based on pixel coordinate extents.

    The function computes the area of each grid cell by calculating the distance between
    the cell boundaries. Two methods are available:
      - High accuracy: uses geodesic distances (via the _measure function).
      - Low accuracy: uses a projected equal-area approximation (EPSG:6933).

    Parameters
    ----------
    input_ds : xr.DataArray or xr.Dataset
         Raster with latitude ('y') and longitude ('x') coordinates in decimal degrees.
    high_accuracy : bool, optional
         If True, use geodesic (great circle) distance calculations.
         If False, use an equal-area projection. If None, a heuristic based on latitude is applied.
    output_in_ha : bool, default True
         If True, converts computed areas from square meters to hectares.

    Returns
    -------
    xr.DataArray
         A DataArray containing the area for each grid cell, with metadata indicating the units and
         the method used (either "geodesic distances" or "EPSG:6933 approximation").
    """
    lat_center = input_ds.y.values  # assumed sorted north to south
    lon_center = input_ds.x.values  # assumed sorted west to east

    if high_accuracy is None:
        high_accuracy = True
        if -70 <= lat_center[0] <= 70:
            high_accuracy = False

    diff_x = np.diff(lon_center)
    diff_y = np.diff(lat_center)
    x_bounds = np.concatenate([[lon_center[0] - diff_x[0] / 2],
                               lon_center[:-1] + diff_x / 2,
                               [lon_center[-1] + diff_x[-1] / 2]])
    y_bounds = np.concatenate([[lat_center[0] - diff_y[0] / 2],
                               lat_center[:-1] + diff_y / 2,
                               [lat_center[-1] + diff_y[-1] / 2]])
    if high_accuracy:
        n_y = len(lat_center)
        n_x = len(lon_center)
        cell_heights = np.array([
            _measure(y_bounds[i], lon_center[0], y_bounds[i+1], lon_center[0])
            for i in range(n_y)
        ])
        y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2
        cell_widths = np.array([
            [_measure(y, x_bounds[j], y, x_bounds[j+1]) for j in range(n_x)]
            for y in y_centers
        ])
        grid_area_m2 = cell_heights[:, None] * cell_widths
    else:
        xv, yv = np.meshgrid(x_bounds, y_bounds, indexing="xy")
        p = Proj("EPSG:6933", preserve_units=False)
        x2, y2 = p(longitude=xv, latitude=yv)
        dx = x2[:-1, 1:] - x2[:-1, :-1]
        dy = y2[1:, :-1] - y2[:-1, :-1]
        grid_area_m2 = np.abs(dx * dy)

    conversion = 1e-4 if output_in_ha else 1.0
    converted_grid_area = grid_area_m2 * conversion
    unit = "ha" if output_in_ha else "m²"
    method = "geodesic distances" if high_accuracy else "EPSG:6933 approximation"

    return xr.DataArray(converted_grid_area,
                        coords={'y': input_ds.y, 'x': input_ds.x},
                        dims=['y', 'x'],
                        attrs={'units': unit,
                               'description': f"Grid cell area in {unit} computed using {method}"})


__all__ = [
    "process_geometry",
    "clip_ds_to_bbox",
    "clip_ds_to_geom",
    "create_area_ds_from_degrees_ds",
    "create_proportion_geom_mask",
    "reproject_match_ds"]

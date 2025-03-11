import json
import warnings
from typing import Union, Optional, List, Protocol

import numpy as np
import xarray as xr
import dask
import dask.array as da
import s3fs
import pyproj
from pyproj import Proj, Geod
from affine import Affine
from rasterio import features
from rasterio.enums import MergeAlg
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

# A simple protocol to type-check geometry-like objects.


class GeometryLike(Protocol):
    geom_type: str


class GeometryData:
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
    Load and validate geometry source.

    Parameters
    ----------
    geom_source : str or list or GeometryLike
         Either a file path (local or S3 URI) to a GeoJSON file, a list of Shapely geometries,
         or a single Shapely geometry.
    dissolve : bool
         If True, all geometries are dissolved into a single geometry.

    Returns
    -------
    tuple
         (geom, geom_crs, geom_bbox)
         - geom: list of Shapely geometries (dissolved if requested)
         - geom_bbox: tuple (minx, miny, maxx, maxy) spanning the geometries
         - geom_area: size of geometry in ha
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
    Compute the area (in square meters) of the supplied geometry.

    Parameters
    ----------
    geom : list
         List of Shapely geometries.
    geom_crs : str
         The source CRS of the geometry.
    target_epsg : int, default=6933
         EPSG code for the target projection.

    Returns
    -------
    float
         Area in square meters.
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


def _measure(lat1, lon1, lat2, lon2) -> float:
    """
    Compute the geodesic distance (in meters) between two points on the WGS84 ellipsoid.
    """
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance


def create_area_ds_from_degrees_ds(input_ds:  Union[xr.DataArray, xr.Dataset],
                                   high_accuracy: Optional[bool] = None,
                                   output_in_ha: bool = True) -> xr.DataArray:
    """
    Calculate the area of each grid cell in a geographic raster and return as a DataArray.

    Parameters
    ----------
    input_ds : xr.DataArray or xr.Dataset
         Input raster with 'y' (latitude) and 'x' (longitude) coordinates (in decimal degrees).
    high_accuracy : bool, optional
         If True, use geodesic calculations; if False, use an equal-area projection.
    output_in_ha : bool, default=True
         If True, convert area from square meters to hectares.

    Returns
    -------
    xr.DataArray
         Grid cell areas in the specified units, with appropriate metadata.
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


def clip_to_bbox(input_ds: Union[xr.DataArray, xr.Dataset], bbox: tuple, drop_time: bool = False) -> xr.DataArray:
    """
    Clip the input raster to the specified bounding box.

    Parameters
    ----------
    input_ds : xr.DataArray or xr.Dataset
         The raster to clip.
    bbox : tuple
         Bounding box (minx, miny, maxx, maxy).
    drop_time : bool, default=False
         If True and a 'time' dimension exists, return the first time slice.

    Returns
    -------
    xr.DataArray
         The spatially subset raster.
    """
    minx, miny, maxx, maxy = bbox
    clipped = input_ds.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    if drop_time and 'time' in clipped.dims:
        return clipped.isel(time=0)
    return clipped


def clip_to_geom(geom_source: ExtendedGeometryInput, input_ds: Union[xr.DataArray, xr.Dataset], all_touch=False) -> xr.DataArray:
    """
    Clip the input raster to the extent of the supplied geometry.

    This function accepts a geometry input that can be either a GeometryData instance or a geometry source (or list of them)
    (e.g. a file path, a single geometry, or a list of geometries). If the provided geom_source is not already a GeometryData,
    it will be processed using process_geometry to obtain the necessary properties (geometry, CRS, and bounding box).

    Parameters
    ----------
    geom_source : ExtendedGeometryInput
         Either a GeometryData instance, a single geometry (or list of geometries), or a file path to a GeoJSON.
         The geometry is expected to have a 'geom_type' attribute if not wrapped in GeometryData.
    input_ds : xr.DataArray or xr.Dataset
         The input raster to be clipped. It must have valid spatial metadata.
    all_touch : bool, optional
         If True, all pixels touched by the geometry will be included. Default is False.

    Returns
    -------
    xr.DataArray
         A raster clipped to the geometry’s extent. Pixels outside the supplied geometry are set to 0 or NaN,
         and the output retains the CRS and spatial properties defined by the geometry source.
    """
    if not isinstance(geom_source, GeometryData):
        geom_source = process_geometry(geom_source, True)
    geom = geom_source.geom
    bbox = geom_source.geom_bbox
    crs = geom_source.geom_crs

    # Prepare spatial subset if needed
    spatial_raster = clip_to_bbox(input_ds, bbox)

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


def align_and_resample_raster(template_raster: Union[xr.DataArray, xr.Dataset],
                              target_raster: Union[xr.DataArray, xr.Dataset],
                              resampling_method="nearest",
                              return_area_grid: bool = True,
                              output_in_ha: bool = True):
    """
    Clip the target raster to the template’s spatial bounds and resample it to match the template’s grid.

    Parameters
    ----------
    template_raster : xr.DataArray or xr.Dataset
         The reference raster whose grid will be used.
    target_raster : xr.DataArray or xr.Dataset
         The raster to be aligned.
    resampling_method : str, optional
         Method for reprojecting (e.g., "nearest", "bilinear").
    return_area_grid : bool, default=True
         If True, compute a grid of cell areas.
    output_in_ha : bool, default=True
         If True, return area in hectares; otherwise in square meters.

    Returns
    -------
    tuple
         (aligned_target, area_target)
    """
    clipped_target = clip_to_bbox(
        template_raster, target_raster.rio.bounds(), drop_time=True)
    aligned_target = clipped_target.rio.reproject_match(
        template_raster, resampling=resampling_method)
    area_target = None
    if return_area_grid:
        area_target = create_area_ds_from_degrees_ds(
            aligned_target, output_in_ha=output_in_ha)
    return aligned_target, area_target


def create_proportion_geom_mask(input_ds: xr.DataArray, pixel_ratio=.001, overwrite=False) -> xr.DataArray:
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
    binary_mask = create_binary_geom_mask(input_ds)

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
    return result


__all__ = [
    "process_geometry",
    "_calculate_geometry_area",
    "create_area_ds_from_degrees_ds",
    "clip_to_bbox",
    "align_and_resample_raster",
    "clip_to_geom"
]

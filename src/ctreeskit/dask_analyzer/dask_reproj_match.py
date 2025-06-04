import numpy as np
import xarray as xr
from typing import Union
from rasterio.enums import Resampling
from .dask_geometry_clip_rio import geometry_clip_rio
import geopandas as gpd
from shapely.geometry import box
from .dask_create_area_ds_from_degrees import create_area_ds_from_degrees_ds_dask
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reproject_match_ds(
    template_raster: Union[xr.DataArray, xr.Dataset],
    target_raster: Union[xr.DataArray, xr.Dataset],
    resampling_method=Resampling.nearest,
    return_area_grid: bool = True,
    output_in_ha: bool = True,
):
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
    target_raster = target_raster.transpose(..., "y", "x")
    template_raster = template_raster.transpose(..., "y", "x")

    bounds = template_raster.rio.bounds()
    print(f"template_raster bounds: {bounds}")
    geom = box(*bounds)
    template_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=template_raster.rio.crs)
    clipped_target = geometry_clip_rio(
        target_raster,
        template_gdf[["geometry"]],
        xdim="x",
        ydim="y",
        all_touched=True,
        invert=False,
    ).isel(time=0)

    print("end clipping to target shape")
    print("start reproject matching")
    aligned_target = clipped_target.rio.reproject_match(
        template_raster, resampling=resampling_method
    )
    print("end reproject matching")
    area_target = None
    print("start create_area_ds_from_degrees_ds")
    if return_area_grid:
        area_target = create_area_ds_from_degrees_ds_dask(
            aligned_target, output_in_ha=output_in_ha
        )
    print("end create_area_ds_from_degrees_ds")
    return aligned_target, area_target

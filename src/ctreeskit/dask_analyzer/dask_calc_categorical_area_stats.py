import xarray as xr
import dask.array as da
from .dask_create_area_ds_from_degrees import create_area_ds_from_degrees_ds_dask
from ctreeskit.xr_analyzer.xr_common import get_single_var_data_array
from ctreeskit.xr_analyzer.xr_zonal_stats_module import (
    _format_output,
    _process_single_var_with_area,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _prepare_area_ds_dask(area_ds, single_var_da):
    logger.info("Preparing area DataArray...")
    """Prepare the area DataArray based on the input type."""
    if "time" in single_var_da.dims:
        template_ds = single_var_da.isel(time=0)
        logger.info("Using first time step as template for area DataArray.")
        logger.info(f"Template DataArray shape: {template_ds.shape}")
    else:
        template_ds = single_var_da
    if area_ds is True:
        return create_area_ds_from_degrees_ds_dask(template_ds)
    if area_ds is False or area_ds is None:
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = da.full(shape, float(1.0), chunks="auto")
        return xr.DataArray(
            const_area,
            coords={"y": template_ds.y.values, "x": template_ds.x.values},
            dims=["y", "x"],
        )
    if isinstance(area_ds, (int, float)):
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = da.full(shape, float(area_ds), chunks="auto")
        return xr.DataArray(
            const_area,
            coords={"y": template_ds.y.values, "x": template_ds.x.values},
            dims=["y", "x"],
        )
    return area_ds


def calculate_categorical_area_stats_dask(
    categorical_ds,
    area_ds=None,
    var_name=None,
    count_name="area_hectares",
    reshape=True,
    drop_zero=True,
    single_class=True,
):
    """Calculate area statistics for each class in categorical raster data."""
    logger.info("Calculating categorical area statistics...")

    logger.info("Getting single variable DataArray...")
    single_var_da = get_single_var_data_array(categorical_ds, var_name)
    logger.info(f"Single variable DataArray: {single_var_da.attrs["flag_meanings"]}")

    logger.info("Preparing area DataArray...")
    area_ds = _prepare_area_ds_dask(area_ds, single_var_da)

    logger.info("Processing single variable with area DataArray...")
    result = _process_single_var_with_area(single_var_da, area_ds, count_name)
    logger.info(f"result columns: {result.columns}")

    logger.info("Formatting output DataFrame...")
    df = _format_output(
        result, single_var_da, count_name, reshape, drop_zero, single_class
    )
    logger.info(f"df columns: {df.columns}")

    return df


__all__ = [
    "_prepare_area_ds_dask",
    "calculate_categorical_area_stats_dask",
]

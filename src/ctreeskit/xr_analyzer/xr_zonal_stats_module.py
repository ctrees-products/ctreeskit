import numpy as np
import xarray as xr
import pandas as pd
from .xr_spatial_processor_module import create_area_ds_from_degrees_ds, reproject_match_ds
from .xr_common import get_single_var_data_array, get_flag_meanings


def calculate_categorical_area_stats(categorical_ds, area_ds=None, var_name=None,
                                     count_name='area_hectares', reshape=True, drop_zero=True,
                                     is_combined=False):
    """
    Calculate area statistics for each class in categorical raster data.

    Works with both time-series and static (non-temporal) rasters.

    Parameters
    ----------
    categorical_ds : xr.Dataset or xr.DataArray
        Categorical raster data (with or without time dimension).
        If Dataset, turns it into dataarray
    area_ds : None, bool, float, or xr.DataArray, optional
        - None: count pixels (area=1.0 per pixel)
        - float/int: constant area per pixel
        - True: calculate area from coordinates
        - DataArray: custom area per pixel
    var_name : str, default None
        Name of the variable in the dataset containing class values
    count_name : str, default "area_hectares"
        Name for the metric column in the output DataFrame
    reshape : bool, default True
        If True, pivots output to wide format with classes as columns
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Results with columns: class values as columns and "total_area"
        For time-series data, time values are included as index
    """
    single_var_da = get_single_var_data_array(categorical_ds, var_name)
    area_ds = _prepare_area_ds(area_ds, single_var_da)
    result = _process_single_var_with_area(single_var_da, area_ds, count_name)
    df = _format_output(result, single_var_da, count_name,
                        reshape, drop_zero, is_combined)
    return df


def _prepare_area_ds(area_ds, single_var_da):
    """Prepare the area DataArray based on the input type."""
    if "time" in single_var_da.dims:
        template_ds = single_var_da.isel(time=0)
    else:
        template_ds = single_var_da
    if area_ds is True:
        return create_area_ds_from_degrees_ds(template_ds)
    if area_ds is False or area_ds is None:
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(1.0))
        return xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                'x': template_ds.x.values},
                            dims=["y", "x"])
    if isinstance(area_ds, (int, float)):
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(area_ds))
        return xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                'x': template_ds.x.values},
                            dims=["y", "x"])

    return area_ds


def _process_single_var_with_area(single_var_da, area_da, count_name):
    """Process the classification DataArray to calculate area statistics into df."""
    result = []
    if "time" in single_var_da.dims:
        for t in single_var_da.time:
            classes_t = single_var_da.sel(time=t)
            sums = (area_da.groupby(classes_t).sum().compute())
            df_t = (sums.to_dataframe(
                count_name).reset_index().assign(time=t.values))
            result.append(df_t)
    else:
        sums = (area_da.groupby(single_var_da).sum().compute())
        df_t = (sums.to_dataframe(count_name).reset_index())
        result.append(df_t)
    return pd.concat(result, ignore_index=True)


def _format_output(df, classification, count_name, reshape, drop_zero,  is_combined=False):
    """Format the output DataFrame."""
    class_var_name = "classification"
    if class_var_name not in df.columns:
        df.rename(columns={df.columns[0]: class_var_name}, inplace=True)
    df = df.astype({count_name: 'float32'})

    if "time" in df.columns:
        d = df[['time', class_var_name, count_name]]
        if reshape:
            d = d.pivot(index='time', columns=class_var_name,
                        values=count_name)
    else:
        d = df[[class_var_name, count_name]]
        if reshape:
            d = d.pivot(columns=class_var_name, values=count_name).iloc[0:1]
            d.index = [0]

    if reshape:
        if not is_combined:
            d = _format_output_reshaped(d, classification, drop_zero)
    return d


def _format_output_reshaped(d, classification, drop_zero=True):
    """
    Reshape and format the output DataFrame.

    Parameters
    ----------
    d : pd.DataFrame
        Already pivoted DataFrame with class values as columns
    classification : xr.DataArray
        Original classification data with potential metadata
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with renamed columns and total area column
    """
    d.columns.name = None
    d.columns = [int(col) if isinstance(col, (int, float))
                 and col.is_integer() else col for col in d.columns]

    flag_meanings = get_flag_meanings(classification)
    d = _rename_columns(d, flag_meanings)
    if drop_zero and 0 in d.columns:
        d = d.drop(columns=[0])
    d['total_area'] = d.sum(axis=1, numeric_only=True)
    return d


def _rename_columns(d, flag_meanings):
    """Rename columns using flag meanings if available."""
    if flag_meanings is not None:
        rename_dict = {}
        for col in d.columns:
            if isinstance(col, (int, np.integer)):
                if 0 <= col-1 < len(flag_meanings):
                    rename_dict[col] = flag_meanings[col-1]
        if rename_dict:
            d = d.rename(columns=rename_dict)
    return d


__all__ = ["calculate_categorical_area_stats"]

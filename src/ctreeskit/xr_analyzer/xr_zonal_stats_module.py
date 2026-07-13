import numpy as np
import xarray as xr
import pandas as pd
from typing import Optional, Union
from xarray.groupers import UniqueGrouper
from .xr_spatial_processor_module import create_area_ds_from_degrees_ds, reproject_match_ds
from .xr_common import (get_single_var_data_array, get_flag_meanings,
                        get_flag_values)


def calculate_categorical_area_stats(
    categorical_ds: Union[xr.Dataset, xr.DataArray],
    area_ds: Optional[Union[bool, float, xr.DataArray]] = None,
    var_name: Optional[str] = None,
    count_name: str = 'area_hectares',
    reshape: bool = True,
    drop_zero: bool = True,
    single_class: bool = True
) -> pd.DataFrame:
    """
    Calculate area statistics for each class in categorical raster data.

    Works with both time-series and static (non-temporal) rasters. Per-class
    totals are computed with a single flag-aware groupby (flox-accelerated,
    dask-compatible): when the data carries CF ``flag_values`` metadata those
    values define the classes, so class labels are matched by value; otherwise
    the unique values present in the data are used.

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
        If True, pivots output to wide format with classes as columns.
        If False, returns a tidy long-format DataFrame.
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Results with columns: class values as columns and "total_area"
        For time-series data, time values are included as index
    """
    single_var_da = get_single_var_data_array(categorical_ds, var_name)
    area_da = _prepare_area_ds(area_ds, single_var_da)
    sums = _sum_area_by_class(single_var_da, area_da, count_name)
    df = _format_output(sums, single_var_da, count_name,
                        reshape, drop_zero, single_class)
    return df


def calculate_combined_categorical_area_stats(primary_ds, secondary_ds, area_ds=None,
                                              count_name='area_hectares', drop_zero=True, reshape=True):
    """
    Calculate area statistics for unique combinations of two categorical datasets and reshape the result
    to include the original classifications and their flags.

    Parameters
    ----------
    primary_ds : xr.DataArray
        First categorical raster dataset.
    secondary_ds : xr.DataArray
        Second categorical raster dataset.
    area_ds : None, bool, float, or xr.DataArray, optional
        - None: count pixels (area=1.0 per pixel)
        - float/int: constant area per pixel
        - True: calculate area from coordinates
        - DataArray: custom area per pixel
    count_name : str, default "area_hectares"
        Name for the metric column in the output DataFrame.
    reshape : bool, default True
        If True, pivots output to wide format with classes as columns.
    drop_zero : bool, default True
        If True, removes combinations where either dataset has a value of 0.

    Returns
    -------
    pd.DataFrame
        Results with columns: original classifications, their flags, and total area.
    """
    matched_secondary, area_grid = reproject_match_ds(primary_ds, secondary_ds)
    if area_ds is None:
        area_ds = area_grid

    combined_classification = create_combined_classification(
        primary_ds, matched_secondary, drop_zero=drop_zero)

    result = calculate_categorical_area_stats(
        combined_classification, area_ds=area_ds, count_name=count_name,
        reshape=True, drop_zero=drop_zero, single_class=False
    )

    if reshape:
        result = _format_output_reshaped_double(
            result, primary_ds, matched_secondary,
            combined_classification.attrs["combined_modulus"], drop_zero)

    return result


def create_combined_classification(primary_ds, secondary_ds, drop_zero=True):
    """Encode two categorical rasters into one combined-class raster.

    Each class pair is packed into a single integer,
    ``primary * modulus + secondary``, where ``modulus`` is the power of ten
    just above the largest secondary class value — so every pair maps to a
    unique code (e.g. modulus 100: primary 3 / secondary 12 -> 312, primary
    4 / secondary 2 -> 402). The modulus is stored in the result's
    ``combined_modulus`` attribute; decode a code with ``divmod(code,
    modulus)``.

    Parameters
    ----------
    primary_ds, secondary_ds : xr.DataArray or single-variable xr.Dataset
        Categorical rasters on the same grid.
    drop_zero : bool, default True
        If True, pixels where either input is 0 are set to 0.

    Returns
    -------
    xr.DataArray
        Integer combined classification with a ``combined_modulus`` attribute.
    """
    primary_da = get_single_var_data_array(primary_ds, None).fillna(0)
    secondary_da = get_single_var_data_array(secondary_ds, None).fillna(0)
    modulus = _pairing_modulus(secondary_da)
    combined = (primary_da.astype("int64") * modulus
                + secondary_da.astype("int64"))
    if drop_zero:
        combined = combined.where(
            (primary_da != 0) & (secondary_da != 0), 0)
    combined.name = "classification"
    combined.attrs["combined_modulus"] = modulus
    return combined


def _pairing_modulus(secondary_da: xr.DataArray) -> int:
    """Power of ten strictly greater than the largest secondary class value."""
    flag_values = get_flag_values(secondary_da)
    if flag_values:
        max_val = max(flag_values)
    else:
        max_val = int(secondary_da.max().compute().item())
    return 10 ** len(str(max(max_val, 1)))


def calculate_stats_with_categories(categorical_da: xr.DataArray,
                                    continuous_da: xr.DataArray,
                                    positive_only: bool = True):
    """
    Calculate statistics for continuous data masked by categories.

    Statistics come from one grouped reduction over the categorical raster
    (flox-accelerated, dask-compatible); categories 0 and NaN are excluded.

    Args:
        categorical_da (xr.DataArray): Categorical mask data
        continuous_da (xr.DataArray): Continuous value data
        positive_only (bool): If True, only continuous values > 0 contribute
            to the statistics. Set False when the variable can legitimately
            be zero or negative (e.g. biomass change).

    Returns:
        pd.DataFrame: Statistics for each category (and time step, if present)
    """
    continuous_matched, _ = reproject_match_ds(
        categorical_da, continuous_da, return_area_grid=False)

    valid = (continuous_matched.where(continuous_matched > 0)
             if positive_only else continuous_matched)
    if "time" in categorical_da.dims and "time" not in valid.dims:
        valid, _ = xr.broadcast(valid, categorical_da)
    valid = valid.assign_coords(category=categorical_da.where(
        categorical_da > 0))

    reduce_dims = [d for d in ("y", "x") if d in valid.dims]
    grouped = valid.groupby(category=UniqueGrouper())
    stats = xr.Dataset({"mean_value": grouped.mean(dim=reduce_dims),
                        "std_value": grouped.std(dim=reduce_dims)}).compute()

    df = stats.to_dataframe().reset_index()
    df["category"] = df["category"].astype(int)
    order = [c for c in ("time", "category") if c in df.columns]
    df = df.sort_values(order).reset_index(drop=True)
    return df[order + ["mean_value", "std_value"]]


def _format_output_reshaped_double(combined_df, primary_ds, secondary_ds,
                                   modulus, drop_zero=True):
    """
    Reshape and format the output DataFrame for two categories.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Already pivoted DataFrame with combined class codes as columns
    primary_ds : xr.DataArray
        First classification data with potential metadata
    secondary_ds : xr.DataArray
        Second classification data with potential metadata
    modulus : int
        Pairing modulus used by ``create_combined_classification``
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with renamed columns and total area column
    """
    combined_df.columns.name = None

    primary_names = _flag_rename_map(_classification_da(primary_ds))
    secondary_names = _flag_rename_map(_classification_da(secondary_ds))

    if drop_zero and 0 in combined_df.columns:
        combined_df = combined_df.drop(columns=[0])

    rename_dict = {}
    for col in combined_df.columns:
        if isinstance(col, (int, np.integer)):
            primary_val, secondary_val = divmod(int(col), modulus)
            meaning_1 = primary_names.get(primary_val, str(primary_val))
            meaning_2 = secondary_names.get(secondary_val, str(secondary_val))
            rename_dict[col] = f"{meaning_1} - {meaning_2}"
    if rename_dict:
        combined_df = combined_df.rename(columns=rename_dict)

    combined_df['total_area'] = combined_df.sum(axis=1, numeric_only=True)
    return combined_df


def _classification_da(ds):
    """The classification DataArray backing a Dataset (or the input itself)."""
    if isinstance(ds, xr.Dataset):
        if "classification" in ds:
            return ds["classification"]
        return get_single_var_data_array(ds, None)
    return ds


def _prepare_area_ds(area_ds, single_var_da):
    """Prepare the area DataArray based on the input type."""
    template_ds = single_var_da.isel(
        time=0) if "time" in single_var_da.dims else single_var_da

    if isinstance(area_ds, bool) and area_ds is True:
        return create_area_ds_from_degrees_ds(template_ds)
    if area_ds is None or area_ds is False:
        # set to pixel count
        area_ds = 1.0
    if isinstance(area_ds, (int, float)):
        return xr.DataArray(
            np.full((template_ds.sizes["y"],
                    template_ds.sizes["x"]), float(area_ds)),
            coords={'y': template_ds.y.values, 'x': template_ds.x.values},
            dims=["y", "x"]
        )
    return area_ds


def _sum_area_by_class(single_var_da, area_da, count_name):
    """Per-class area totals from one grouped sum over the class raster.

    Returns a DataArray named ``count_name`` with a ``classification``
    dimension (plus ``time`` when present). Classes in the label set that do
    not occur at a given time step get an explicit 0.0.
    """
    weights = xr.ones_like(single_var_da, dtype="float64") * area_da
    weights.name = count_name
    weights = weights.assign_coords(classification=single_var_da)

    labels = _groupby_labels(single_var_da)
    reduce_dims = [d for d in ("y", "x") if d in weights.dims]
    sums = (weights
            .groupby(classification=UniqueGrouper(labels=labels))
            .sum(dim=reduce_dims)
            .compute()
            .fillna(0.0))

    class_values = np.asarray(sums["classification"].values)
    if class_values.dtype.kind == "f" and np.all(np.mod(class_values, 1) == 0):
        sums = sums.assign_coords(classification=class_values.astype(int))
    return sums


def _groupby_labels(single_var_da):
    """Class labels for the grouper.

    CF ``flag_values`` metadata defines the label set when present (0 is
    always included so ``drop_zero`` has a column to act on). Otherwise the
    labels are the unique values in the data; lazy (dask) input requires an
    explicit label set, so uniques are computed up front in that case.
    """
    flag_values = get_flag_values(single_var_da)
    if flag_values is not None:
        return sorted(set(flag_values) | {0})
    data = single_var_da.data
    if getattr(data, "chunks", None) is not None:
        import dask.array as dask_array
        uniques = np.asarray(dask_array.unique(data).compute())
        return [v for v in uniques.tolist() if not np.isnan(v)]
    return None


def _format_output(sums, classification_ds, count_name, reshape, drop_zero,
                   single_class=True):
    """Convert per-class totals to the output DataFrame."""
    if not reshape:
        df = sums.to_dataframe().reset_index()
        columns = [c for c in ("time", "classification", count_name)
                   if c in df.columns]
        return df[columns]

    if "time" in sums.dims:
        result_df = sums.transpose("time", "classification").to_pandas()
    else:
        result_df = sums.to_pandas().to_frame().T
        result_df.index = [0]
    result_df.columns.name = None

    if single_class:
        result_df = _format_output_reshaped(
            result_df, classification_ds, drop_zero)
    return result_df


def _format_output_reshaped(input_df, classification_ds, drop_zero=True):
    """
    Reshape and format the output DataFrame.

    Parameters
    ----------
    input_df : pd.DataFrame
        Already pivoted DataFrame with class values as columns
    classification_ds : xr.DataArray
        Original classification data with potential metadata
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with renamed columns and total area column
    """
    if drop_zero and 0 in input_df.columns:
        input_df = input_df.drop(columns=[0])
    rename_map = _flag_rename_map(classification_ds)
    columns_to_rename = {col: rename_map[col] for col in input_df.columns
                         if col in rename_map}
    if columns_to_rename:
        input_df = input_df.rename(columns=columns_to_rename)
    input_df['total_area'] = input_df.sum(axis=1, numeric_only=True)
    return input_df


def _flag_rename_map(classification_ds):
    """Map class values to flag meanings, matched by value.

    Pairs CF ``flag_values`` with ``flag_meanings``. When only
    ``flag_meanings`` is present, meanings are assigned to values 1..N in
    order.
    """
    flag_meanings = get_flag_meanings(classification_ds)
    if flag_meanings is None:
        return {}
    flag_values = get_flag_values(classification_ds)
    if flag_values is None or len(flag_values) != len(flag_meanings):
        flag_values = range(1, len(flag_meanings) + 1)
    return dict(zip(flag_values, flag_meanings))


__all__ = ["calculate_categorical_area_stats",
           "calculate_combined_categorical_area_stats",
           "create_combined_classification"]

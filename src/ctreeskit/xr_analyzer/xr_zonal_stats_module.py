import numpy as np
import xarray as xr
import pandas as pd
from typing import Optional, Union
from .xr_spatial_processor_module import create_area_ds_from_degrees_ds, reproject_match_ds
from .xr_common import get_single_var_data_array, get_flag_meanings


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
                        reshape, drop_zero, single_class)
    return df


def calculate_combined_categorical_area_stats(primary_ds, secondary_ds, area_ds=None,
                                              count_name='area_hectares', drop_zero=True, reshape=True):
    """
    Calculate area statistics for unique combinations of two categorical datasets and reshape the result
    to include the original classifications and their flags.

    Parameters
    ----------
    categorical_ds1 : xr.DataArray
        First categorical raster dataset.
    categorical_ds2 : xr.DataArray
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
    matched_secondary, area_ds = reproject_match_ds(primary_ds, secondary_ds)

    combined_classification = create_combined_classification(
        primary_ds, matched_secondary)

    # Run the existing calculate_categorical_area_stats function
    result = calculate_categorical_area_stats(
        combined_classification, area_ds=area_ds, count_name=count_name,
        reshape=True, drop_zero=True, single_class=False
    )

    if reshape:
        result = _format_output_reshaped_double(
            result, primary_ds, matched_secondary, drop_zero)

    return result


def create_combined_classification(primary_ds, secondary_ds, drop_zero=True):
    # Combine the two datasets into a single "combined classification" as a float
    combined_classification = primary_ds.astype(
        float) + (secondary_ds.astype(float) / 10)
    # Optionally drop combinations where either dataset has a value of 0
    if drop_zero:
        combined_classification = combined_classification.where(
            (secondary_ds != 0) & (primary_ds != 0), 0
        )
    return combined_classification


def calculate_stats_with_categories(categorical_da: xr.DataArray,
                                    continuous_da: xr.DataArray):
    """
    Calculate statistics for continuous data masked by categories.

    Args:
        categorical_da (xr.DataArray): Categorical mask data
        continuous_da (xr.DataArray): Continuous value data

    Returns:
        List[Dict]: Statistics for each category
    """
    # Get unique categories (excluding 0 and NaN)
    continuous_matched, _ = reproject_match_ds(
        categorical_da, continuous_da, return_area_grid=False)
   # Initialize results dictionary

    # Check if the categorical DataArray has a time dimension
    if "time" in categorical_da.dims:
        # Iterate over each time step
        for t in categorical_da.time:
            # Select the data for the current time step
            categorical_t = categorical_da.sel(time=t)
            continuous_t = continuous_matched.sel(time=t)
            results = _calculate_cont_cat_stats(categorical_t, continuous_t)
            results["time"].append(t.values)
    else:
        results = _calculate_cont_cat_stats(categorical_da, continuous_matched)
    return pd.DataFrame(results)


def _calculate_cont_cat_stats(categorical_da, continuous_da):
    categories = categorical_da.where(categorical_da > 0).unique().values
    categories = categories[~np.isnan(categories)].astype(float)
    results = {
        "time": [],
        "category": [],
        "mean_value": [],
        "std_value": []
    }
    # Calculate statistics for each category
    for category in categories:
        # Create mask for the current category
        mask = categorical_da == category

        # Mask the continuous data for the current category
        masked_values = continuous_da.where(
            mask & (continuous_da > 0))

        # Calculate mean and standard deviation, handling empty data
        if masked_values.count() > 0:
            mean_value = float(masked_values.mean())
            std_value = float(masked_values.std())
        else:
            mean_value = None
            std_value = None
            # Append results
        results["category"].append(int(category))
        results["mean_value"].append(mean_value)
        results["std_value"].append(std_value)
    return results


def _format_output_reshaped_double(combined_df, primary_ds, secondary_ds, drop_zero=True):
    """
    Reshape and format the output DataFrame for two categories.

    Parameters
    ----------
    combined_df : pd.DataFrame
        Already pivoted DataFrame with class values as columns
    primary_ds : xr.DataArray
        First classification data with potential metadata
    cat_2 : xr.DataArray
        Second classification data with potential metadata
    secondary_ds : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with renamed columns and total area column
    """
    combined_df.columns.name = None

    # Split column names by "." and map to flag meanings
    primary_flag_meanings = get_flag_meanings(primary_ds.classification)
    seconday_flag_meanings = get_flag_meanings(secondary_ds.classification)
    rename_dict = {}
    for col in combined_df.columns:
        if (isinstance(col, str) and "." in col) or isinstance(col, float):
            col = str(col)
            # Split the column name into two parts
            part_1, part_2 = col.split(".")
            try:
                # Map each part to its respective flag meaning
                meaning_1 = primary_flag_meanings[int(
                    part_1) - 1] if primary_flag_meanings and int(part_1) - 1 < len(primary_flag_meanings) else part_1
                meaning_2 = seconday_flag_meanings[int(
                    part_2) - 1] if seconday_flag_meanings and int(part_2) - 1 < len(seconday_flag_meanings) else part_2
                rename_dict[col] = f"{meaning_1} - {meaning_2}"
            except (ValueError, IndexError):
                # If mapping fails, keep the original column name
                rename_dict[col] = col

    # Drop zero column if requested
    if drop_zero and (0 in combined_df.columns or "0.0" in combined_df.columns):
        combined_df = combined_df.drop(
            columns=[col for col in [0, "0.0"] if col in combined_df.columns])

    # Rename columns using the constructed dictionary
    if rename_dict:
        combined_df.columns = combined_df.columns.map(str)
        combined_df = combined_df.rename(columns=rename_dict)

    # Add total area column
    combined_df['total_area'] = combined_df.sum(axis=1, numeric_only=True)

    return combined_df


def _prepare_area_ds(area_ds, single_var_da):
    """Prepare the area DataArray based on the input type."""
    template_ds = single_var_da.isel(
        time=0) if "time" in single_var_da.dims else single_var_da

    if isinstance(area_ds, bool) and area_ds is True:
        return create_area_ds_from_degrees_ds(template_ds)
    if area_ds in [False, None]:
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


def _format_output(input_df, classification_ds, count_name, reshape, drop_zero, single_class=True):
    """Format the output DataFrame."""
    class_var_name = "classification"
    if class_var_name not in input_df.columns:
        input_df.rename(
            columns={input_df.columns[0]: class_var_name}, inplace=True)
    input_df = input_df.astype({count_name: 'float32'})

    if "time" in input_df.columns:
        result_df = input_df[['time', class_var_name, count_name]]
        if reshape:
            result_df = result_df.pivot(index='time', columns=class_var_name,
                                        values=count_name)
    else:
        result_df = input_df[[class_var_name, count_name]]
        if reshape:
            result_df = result_df.pivot(
                columns=class_var_name, values=count_name).iloc[0:1]
            result_df.index = [0]

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
    classification : xr.DataArray
        Original classification data with potential metadata
    drop_zero : bool, default True
        If True, removes class 0 (typically no-data) from results

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with renamed columns and total area column
    """
    input_df.columns.name = None
    input_df.columns = [int(col) if isinstance(col, (int, float))
                        and col.is_integer() else col for col in input_df.columns]

    flag_meanings = get_flag_meanings(classification_ds)
    input_df = _rename_columns(input_df, flag_meanings)
    if drop_zero and 0 in input_df.columns:
        input_df = input_df.drop(columns=[0])
    input_df['total_area'] = input_df.sum(axis=1, numeric_only=True)
    return input_df


def _rename_columns(input_df, flag_meanings):
    """Rename columns using flag meanings if available."""
    if flag_meanings is not None:
        rename_dict = {}
        for col in input_df.columns:
            if isinstance(col, (int, np.integer)):
                if 0 <= col-1 < len(flag_meanings):
                    rename_dict[col] = flag_meanings[col-1]
        if rename_dict:
            input_df = input_df.rename(columns=rename_dict)
    return input_df


__all__ = ["calculate_categorical_area_stats",
           "calculate_combined_categorical_area_stats",
           "create_combined_classification"]

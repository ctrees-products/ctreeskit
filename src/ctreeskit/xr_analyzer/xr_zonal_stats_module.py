import numpy as np
import xarray as xr
import pandas as pd
from .xr_spatial_processor_module import create_area_ds_from_degrees_ds
import cf_xarray


def calculate_categorical_area_stats(categorical_ds, area_ds=None, var_name=None,
                                     count_name='area_hectares', reshape=True, drop_zero=True):
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

    classification = None
    if isinstance(categorical_ds, xr.DataArray):
        # If input is already a DataArray, use it directly
        classification = categorical_ds
    else:
        # If input is a Dataset
        if var_name is not None:
            # Use the specified variable if provided
            classification = categorical_ds[var_name]
        else:
            # Try to automatically determine the variable
            data_vars = list(categorical_ds.data_vars)
            if len(data_vars) == 1:
                # If there's only one variable, use it
                classification = categorical_ds[data_vars[0]]
            else:
                raise ValueError(
                    f"Dataset has multiple variables ({data_vars}). "
                    "Please specify 'var_name' parameter."
                )
    # Force zeros to remain zero.
    classification = classification.where(classification != 0, 0)
    if "time" in classification.dims:
        template_ds = classification.isel(time=0)
    else:
        template_ds = classification

    # Based on the type of area_ds, prepare the area dataset.
    if isinstance(area_ds, (int, float)):
        # Create a constant area DataArray matching the spatial dims
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(area_ds))
        area_ds = xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                   'x': template_ds.x.values},
                               dims=["y", "x"])
    elif isinstance(area_ds, bool):
        # Create a constant area DataArray with pixel area from coordinates.
        area_ds = create_area_ds_from_degrees_ds(template_ds)
    elif area_ds is None:
        # just results pixel count
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(1.0))
        area_ds = xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                   'x': template_ds.x.values},
                               dims=["y", "x"])

    # Process each time step individually
    result = []
    if "time" in classification.dims:
        # Process each time step individually for time series data
        for t in classification.time:
            # Get classification values for this time step
            classes_t = classification.sel(time=t)

            # Group areas by class values
            sums = (area_ds
                    .groupby(classes_t)
                    .sum()
                    .compute())  # Single compute call per time step

            # Convert to DataFrame with time information
            df_t = (sums
                    .to_dataframe(count_name)
                    .reset_index()
                    .assign(time=t.values))

            result.append(df_t)
    else:
        # Process non-time series data (single step)
        sums = (area_ds
                .groupby(classification)
                .sum()
                .compute())  # Single compute call

        # Convert to DataFrame (no time information)
        df_t = (sums
                .to_dataframe(count_name)
                .reset_index())

        result.append(df_t)

    # Combine all results
    df = pd.concat(result, ignore_index=True)
    class_var_name = "classification"
    # Ensure the class_var_name is in the DataFrame columns
    if class_var_name not in df.columns:
        df.rename(columns={df.columns[0]: class_var_name}, inplace=True)

    # Apply dtype conversion
    df = df.astype({
        class_var_name: 'int32',
        count_name: 'float32'
    })

    # Format the output
    if "time" in df.columns:
        d = df[['time', class_var_name, count_name]]
        if reshape:
            d = d.pivot(index='time', columns=class_var_name,
                        values=count_name)
    else:
        d = df[[class_var_name, count_name]]
        if reshape:
            d = d.pivot(columns=class_var_name, values=count_name).iloc[0:1]
            d.index = [0]  # Set a simple index for non-time series data

    if reshape:
        d = reshape_output(d, classification, drop_zero)
    return d


def reshape_output(d, classification, drop_zero=True):
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
    # Remove column name from pivoted DataFrame
    d.columns.name = None

    # Get the classification data - flag meanings
    flag_meanings = None
    try:
        # Try to get flag metadata from attributes
        if hasattr(classification, 'attrs') and 'flag_meanings' in classification.attrs:
            flag_meanings = classification.attrs['flag_meanings'].split()
    except Exception:
        # If anything goes wrong, just continue without flag names
        pass

    # Rename columns using flag meanings if available
    if flag_meanings is not None:
        rename_dict = {}
        for col in d.columns:
            if isinstance(col, (int, np.integer)):
                # Check if we have a flag meaning for this value (adjust index if needed)
                if 0 <= col-1 < len(flag_meanings):
                    rename_dict[col] = flag_meanings[col-1]

        # Apply the renaming if we found any matches
        if rename_dict:
            d = d.rename(columns=rename_dict)

    if drop_zero and 0 in d.columns:
        # Drop any zero columns if they exist
        d = d.drop(columns=[0])

    # Add total area column
    d['total_area'] = d.sum(axis=1, numeric_only=True)

    return d


__all__ = ["calculate_categorical_area_stats"]

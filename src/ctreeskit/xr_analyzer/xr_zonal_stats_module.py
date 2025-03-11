import numpy as np
import xarray as xr
import pandas as pd
from xr_spatial_processor_module import create_area_ds_from_degrees_ds


def calculate_categorical_area_stats(categorical_ds, area_ds=None, classification_values=None):
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
    classification_values : list, optional
        List of class values to analyze. Default uses unique values from data.

    Returns
    -------
    pd.DataFrame
        Results with columns: "0" (total area), "1"..."n" (per-class areas),
        and "time" (if input has time dimension).
    """
    # Prepare the classification dataset: force zeros to remain zero
    # and convert to dataarray.
    if isinstance(categorical_ds, xr.Dataset):
        categorical_ds = categorical_ds.to_datarray()

    # Force zeros to remain zero.
    categorical_ds = categorical_ds.where(categorical_ds != 0, 0)
    if "time" in categorical_ds.dims:
        template_ds = categorical_ds.isel(time=0)
    else:
        template_ds = categorical_ds

    # Based on the type of area_ds, prepare the area dataset.
    if isinstance(area_ds, (int, float)):
        # Create a constant area DataArray matching the spatial dims
        # Create a constant area DataArray matching the spatial dims.
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(area_ds))
        area_ds = xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                   'x': template_ds.x.values},
                               dims=["y", "x"])
    elif isinstance(area_ds, bool):
        # Create a constant area DataArray with pixel area from coordinates.
        area_ds = create_area_ds_from_degrees_ds(categorical_ds.isel(time=0))
    elif area_ds is None:
        # just results pixel count
        # Create a constant area DataArray matching the spatial dims.
        shape = (template_ds.sizes["y"], template_ds.sizes["x"])
        const_area = np.full(shape, float(1.0))
        area_ds = xr.DataArray(const_area, coords={'y': template_ds.y.values,
                                                   'x': template_ds.x.values},
                               dims=["y", "x"])
    # Define the class values
    if classification_values:
        cl_values = list(classification_values)
    else:
        # If none provided, use sorted unique values from the first time slice.
        cl_values = sorted(np.unique(categorical_ds.isel(time=0).values))

    if "time" not in categorical_ds.dims:
        return _calculate_area_stats(cl_values, categorical_ds, area_ds)
    else:
        # Initialize an empty list for DataFrames (one per time slice).
        time_results = []

        # Loop over each time slice in the classification dataset.
        for t in categorical_ds.time.values:
            da_class_t = categorical_ds.sel(time=t)
            df_t = _calculate_area_stats(cl_values, da_class_t, area_ds)
            df_t["time"] = t
            time_results.append(df_t)

        # Concatenate all time slice DataFrames into one DataFrame.
        df_all = pd.concat(time_results, ignore_index=True)
        return df_all


def _calculate_area_stats(cl_values, da_class_t, area_ds):
    """
    Helper function to calculate area statistics for a single time slice.

    Parameters
    ----------
    cl_values : list
        List of class values to analyze
    da_class_t : xr.DataArray
        DataArray containing class values for a single time slice
    area_ds : xr.DataArray
        DataArray containing area values per pixel

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with total area in column "0" and
        class areas in columns "1"..."n"
    """
    n_classes = len(cl_values)
    # Array to store per-class area (ha)
    t_area_class = np.zeros(n_classes)

    # Loop over each classification class.
    for i in range(n_classes):
        # Create a mask for pixels corresponding to the current class,
        # multiply by the pixel area, and sum up the areas.
        area_class = da_class_t.where(
            da_class_t == cl_values[i], 0) * area_ds
        t_area_class[i] = np.nansum(area_class.values)

    # Compute the overall total area.
    area_total_ha = np.nansum(t_area_class)

    # Concatenate total area and per-class areas into one array.
    data = np.concatenate([np.array([area_total_ha]), t_area_class])

    # Use simple numeric column names as strings.
    col_names = [str(i) for i in range(len(data))]
    return pd.DataFrame(data.reshape(1, -1), columns=col_names)


__all__ = [
    "calculate_categorical_area_stats"]

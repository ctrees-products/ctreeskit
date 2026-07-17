import xarray as xr


def get_single_var_data_array(xr_dataset, var_name):
    """Get the single DataArray from the input dataset."""
    if isinstance(xr_dataset, xr.DataArray):
        return xr_dataset
    if var_name is not None:
        return xr_dataset[var_name]
    data_vars = list(xr_dataset.data_vars)
    if len(data_vars) == 1:
        return xr_dataset[data_vars[0]]
    raise ValueError(
        f"Dataset has multiple variables ({data_vars}). "
        "Please specify 'var_name' parameter."
    )


def get_flag_meanings(xr_dataset):
    """Get flag meanings from the dataset attributes."""
    try:
        if hasattr(xr_dataset, "attrs") and "flag_meanings" in xr_dataset.attrs:
            return xr_dataset.attrs["flag_meanings"].split()
    except Exception:
        pass
    return None


def get_flag_values(xr_dataset):
    """Get integer flag values from the dataset attributes.

    Accepts either a sequence (CF-style ``flag_values`` array) or a
    space-separated string, returning a list of ints, or None when the
    attribute is absent.
    """
    attrs = getattr(xr_dataset, "attrs", {})
    flag_values = attrs.get("flag_values")
    if flag_values is None:
        return None
    if isinstance(flag_values, str):
        flag_values = flag_values.split()
    return [int(v) for v in flag_values]


def agg_classified_mapped_da(classification_mapping: dict, data_array: xr.DataArray):
    """
    Aggregate classification mapping to a DataArray.

    Args:
        classification_mapping (dict): Mapping of class names to class ids.

    Returns:
        xr.DataArray: DataArray with aggregated class ids.
    """
    classes = []
    for key, value in classification_mapping.items():
        if len(value) == 1:
            classes.append(value[0])
        else:
            data_array = data_array.where(
                ~data_array.isin(value), value[0]
            )  ## this is the modified data array and should be returned
            classes.append(value[0])
    return (classes, data_array)


__all__ = ["get_single_var_data_array", "get_flag_meanings",
           "get_flag_values", "agg_classified_mapped_da"]

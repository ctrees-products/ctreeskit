"""Turn categorical or boolean raster layers into vector patches and events.

The workflow this module supports is:

1. ``patches_from_mask`` / ``patches_from_categorical`` / ``patches_over_time``
   polygonize connected pixel regions into a :class:`geopandas.GeoDataFrame`,
   one row per region, carrying a true-area column in hectares.
2. ``merge_patches`` groups patches that lie within a metric distance of one
   another into larger *events*, returning the events and a membership table.
3. ``assign_to_polygons`` relates events to reference polygons with a
   many-to-many overlap table.
4. ``write_geoparquet`` / ``read_geoparquet`` round-trip either table.

Vector output depends on geopandas (and pyarrow for GeoParquet), which ship in
the optional ``vector`` extra: ``pip install 'ctreeskit[vector]'``. The imports
are deferred, so importing this module without the extra succeeds and only the
functions that need it raise.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import pyproj
import shapely
import xarray as xr
from rasterio import features
from shapely.geometry import shape

from .xr_spatial_processor_module import create_area_ds_from_degrees_ds

if TYPE_CHECKING:  # pragma: no cover - typing only
    from geopandas import GeoDataFrame

# Equal-area projection used for true-area measurement of vector geometries,
# matching the projection the spatial processor uses for geometry areas.
EQUAL_AREA_EPSG = 6933

M2_TO_HA = 1e-4


def _require_geopandas():
    """Import geopandas, raising a directed ImportError when it is missing.

    Raises
    ------
    ImportError
        If geopandas is not installed. It ships with the optional ``vector``
        extra: ``pip install 'ctreeskit[vector]'``.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "geopandas is required for vector patch output. Install the "
            "'vector' extra: pip install 'ctreeskit[vector]'"
        ) from e
    return gpd


def _require_pyarrow() -> None:
    """Import pyarrow, raising a directed ImportError when it is missing."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyarrow is required to read and write GeoParquet. Install the "
            "'vector' extra: pip install 'ctreeskit[vector]'"
        ) from e


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Return the (y, x) dimension names rioxarray reports for a raster."""
    try:
        return da.rio.y_dim, da.rio.x_dim
    except Exception:
        if "y" in da.dims and "x" in da.dims:
            return "y", "x"
        raise ValueError(
            "Could not determine the spatial dimensions. Set them with "
            "`da.rio.set_spatial_dims(x_dim=..., y_dim=...)`."
        )


def _pixel_area_ha(mask: xr.DataArray, crs) -> np.ndarray:
    """Per-pixel area in hectares for the grid ``mask`` sits on.

    Geographic (lat/lon) grids are measured cell by cell with
    ``create_area_ds_from_degrees_ds``, so cells shrink poleward; degrees are
    never treated as a linear unit. Projected grids use the constant pixel area
    implied by the affine transform, after checking the CRS is metre-based.

    Returns
    -------
    numpy.ndarray
        A ``(y, x)`` array of hectares per pixel.
    """
    y_dim, x_dim = _spatial_dims(mask)
    crs = pyproj.CRS.from_user_input(crs)
    if crs.is_geographic:
        renamed = mask.rename({y_dim: "y", x_dim: "x"})
        return np.asarray(
            create_area_ds_from_degrees_ds(renamed, output_in_ha=True).values
        )

    unit = {axis.unit_name for axis in crs.axis_info}
    if not unit <= {"metre", "meter", "m"}:
        raise ValueError(
            f"Projected CRS axis units are {sorted(unit)}; area in hectares is "
            "only computed for metre-based projected CRSs or geographic CRSs."
        )
    transform = mask.rio.transform()
    cell_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    return np.full(mask.shape, cell_m2 * M2_TO_HA, dtype="float64")


def patches_from_mask(
    mask: xr.DataArray,
    *,
    connectivity: int = 8,
    min_area_ha: float | None = None,
    fill_value: Any = None,
    attrs: dict | None = None,
    area_ha_grid: np.ndarray | None = None,
) -> "GeoDataFrame":
    """Polygonize connected regions of a raster mask into patches.

    Parameters
    ----------
    mask : xr.DataArray
        A 2-D boolean or integer raster carrying a rioxarray CRS and transform.
        Non-zero (or True) pixels are polygonized; zero/False pixels are
        background. Geographic (lat/lon) and projected grids are both supported.
    connectivity : int, default 8
        Pixel connectivity used to group pixels into regions: 4 (edge-sharing
        only) or 8 (edges and corners).
    min_area_ha : float, optional
        Minimum mapping unit. Regions whose true area is below this many
        hectares are dropped. The threshold is applied to measured area, not to
        pixel counts, so it stays correct on lat/lon grids where pixel size
        varies with latitude.
    fill_value : optional
        Value marking nodata in ``mask``. Pixels equal to it are treated as
        background. Leave as None when passing an already-clean boolean mask, or
        when nodata is already encoded as False/0.
    attrs : dict, optional
        Constant columns broadcast onto every row, for example a layer name, a
        class value, or a time step.
    area_ha_grid : numpy.ndarray, optional
        Per-pixel area in hectares for the grid ``mask`` sits on, shaped like
        ``mask``. Computed from the CRS and transform when omitted; pass it to
        reuse one measurement across many layers on the same grid.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per region, with columns ``patch_id`` (sequential and stable
        within the call), ``pixel_count``, ``area_ha``, ``geometry``, plus any
        ``attrs`` keys. The CRS is the raster's CRS.

    Raises
    ------
    ValueError
        If ``mask`` is not 2-D, has no CRS, or ``connectivity`` is not 4 or 8.
    """
    gpd = _require_geopandas()

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if mask.ndim != 2:
        raise ValueError(
            f"mask must be 2-D; got dims {mask.dims}. Select a single slice first."
        )
    crs = mask.rio.crs
    if crs is None:
        raise ValueError(
            "mask has no CRS. Set one with `mask.rio.write_crs(...)` first."
        )

    values = np.asarray(mask.values)
    bool_mask = values != 0
    if fill_value is not None:
        bool_mask &= values != fill_value
    bool_mask = np.ascontiguousarray(bool_mask)

    transform = mask.rio.transform()
    geometries = [
        shape(geom)
        for geom, _ in features.shapes(
            bool_mask.astype("uint8"),
            mask=bool_mask,
            connectivity=connectivity,
            transform=transform,
        )
    ]

    columns: dict[str, Any] = {
        "patch_id": np.arange(1, len(geometries) + 1, dtype="int64"),
        "pixel_count": np.zeros(len(geometries), dtype="int64"),
        "area_ha": np.zeros(len(geometries), dtype="float64"),
    }

    if geometries:
        # Burn the polygons back onto the grid so each pixel carries its
        # patch id, then reduce pixel counts and areas per id in one pass.
        labels = features.rasterize(
            ((geom, i) for i, geom in enumerate(geometries, start=1)),
            out_shape=bool_mask.shape,
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        flat = labels.ravel()
        n_bins = len(geometries) + 1
        columns["pixel_count"] = np.bincount(flat, minlength=n_bins)[1:].astype("int64")
        if area_ha_grid is None:
            area_ha_grid = _pixel_area_ha(mask, crs)
        elif np.shape(area_ha_grid) != bool_mask.shape:
            raise ValueError(
                f"area_ha_grid shape {np.shape(area_ha_grid)} does not match the "
                f"mask shape {bool_mask.shape}."
            )
        columns["area_ha"] = np.bincount(
            flat, weights=area_ha_grid.ravel(), minlength=n_bins
        )[1:]

    if attrs:
        for key, value in attrs.items():
            columns[key] = [value] * len(geometries)

    patches = gpd.GeoDataFrame(columns, geometry=geometries, crs=crs)

    if min_area_ha is not None:
        patches = patches[patches["area_ha"] >= min_area_ha]
        patches = patches.reset_index(drop=True)
        patches["patch_id"] = np.arange(1, len(patches) + 1, dtype="int64")
    return patches


def patches_from_categorical(
    da: xr.DataArray,
    values: int | Sequence[int],
    **kwargs,
) -> "GeoDataFrame":
    """Polygonize the pixels of a categorical raster matching given class values.

    Parameters
    ----------
    da : xr.DataArray
        A 2-D categorical raster with a rioxarray CRS and transform.
    values : int or sequence of int
        Class value(s) to select. Pixels matching any of them form the mask.
    **kwargs
        Passed through to :func:`patches_from_mask` (``connectivity``,
        ``min_area_ha``, ``attrs``, ``area_ha_grid``).

    Returns
    -------
    geopandas.GeoDataFrame
        Patches as returned by :func:`patches_from_mask`, with an extra
        ``class_values`` column recording the matched value(s) as a
        comma-separated string.
    """
    value_list: list[int] = (
        [int(values)] if isinstance(values, (int, np.integer)) else [int(v) for v in values]
    )
    mask = da.isin(value_list)
    mask.rio.write_crs(da.rio.crs, inplace=True)
    mask.rio.write_transform(da.rio.transform(), inplace=True)

    attrs = dict(kwargs.pop("attrs", None) or {})
    attrs.setdefault("class_values", ",".join(str(v) for v in value_list))
    return patches_from_mask(mask, attrs=attrs, **kwargs)


def patches_over_time(
    da: xr.DataArray,
    values: int | Sequence[int],
    *,
    time_dim: str = "time",
    **kwargs,
) -> "GeoDataFrame":
    """Polygonize a categorical raster one time step at a time.

    Each step is selected and materialized on its own, so a lazily backed cube
    is never loaded in full. The per-pixel area grid is measured once and shared
    by every step.

    Parameters
    ----------
    da : xr.DataArray
        A categorical raster with a time dimension, a rioxarray CRS and a
        transform.
    values : int or sequence of int
        Class value(s) to select, as in :func:`patches_from_categorical`.
    time_dim : str, default "time"
        Name of the dimension to iterate.
    **kwargs
        Passed through to :func:`patches_from_categorical`.

    Returns
    -------
    geopandas.GeoDataFrame
        All steps concatenated, with a ``time`` column holding the coordinate
        value of the step each patch came from (a :class:`pandas.Timestamp`
        for datetime coordinates) and ``patch_id`` unique across the whole
        result.

    Raises
    ------
    ValueError
        If ``time_dim`` is not a dimension of ``da``.
    """
    gpd = _require_geopandas()

    if time_dim not in da.dims:
        raise ValueError(f"'{time_dim}' is not a dimension of the input ({da.dims}).")

    crs = da.rio.crs
    if crs is None:
        raise ValueError("input has no CRS. Set one with `da.rio.write_crs(...)` first.")
    transform = da.rio.transform()
    base_attrs = dict(kwargs.pop("attrs", None) or {})
    area_ha_grid = kwargs.pop("area_ha_grid", None)
    frames = []
    for i in range(da.sizes[time_dim]):
        step = da.isel({time_dim: i})
        step = step.compute()
        step.rio.write_crs(crs, inplace=True)
        step.rio.write_transform(transform, inplace=True)
        if area_ha_grid is None:
            area_ha_grid = _pixel_area_ha(step, crs)
        attrs = dict(base_attrs)
        attrs["time"] = _coord_scalar(step[time_dim]) if time_dim in step.coords else i
        frames.append(patches_from_categorical(
            step, values, attrs=attrs, area_ha_grid=area_ha_grid, **kwargs))

    if not frames:
        return gpd.GeoDataFrame(
            {"patch_id": [], "pixel_count": [], "area_ha": [], "time": []},
            geometry=[],
            crs=crs,
        )

    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=crs)
    out["patch_id"] = np.arange(1, len(out) + 1, dtype="int64")
    return out


def _coord_scalar(coord: xr.DataArray) -> Any:
    """A 0-d coordinate as a Python-friendly scalar.

    Datetime coordinates become :class:`pandas.Timestamp` whatever their
    precision; other dtypes go through ``.item()``.
    """
    value = coord.values
    if value.dtype.kind == "M":
        # .item() on a datetime64[ns] scalar is an int of nanoseconds since the
        # epoch, which is exactly what pd.Timestamp takes.
        return pd.Timestamp(value.astype("datetime64[ns]").item())
    return coord.item()


def _metric_crs(patches: "GeoDataFrame", metric_crs=None):
    """Pick a CRS in which buffering by metres is meaningful.

    A geographic input is projected to its estimated UTM zone, which preserves
    local distance far better than an equal-area projection would.
    """
    if metric_crs is not None:
        return pyproj.CRS.from_user_input(metric_crs)
    crs = patches.crs
    if crs is None:
        raise ValueError("patches has no CRS; set one before merging.")
    if crs.is_geographic:
        return patches.estimate_utm_crs()
    return crs


def _connected_components(geoms, n: int) -> np.ndarray:
    """Label geometries so that any two that intersect share a label."""
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    for a, b in zip(left, right):
        if a != b:
            union(int(a), int(b))
    return np.array([find(i) for i in range(n)])


def merge_patches(
    patches: "GeoDataFrame",
    *,
    gap_m: float,
    group_by: Sequence[str] | None = None,
    metric_crs=None,
) -> tuple["GeoDataFrame", pd.DataFrame]:
    """Group patches separated by no more than ``gap_m`` metres into events.

    Each patch is buffered by ``gap_m / 2`` in a metric CRS; patches whose
    buffers intersect fall in the same event. Buffering happens in a projected
    CRS — the estimated UTM zone when the input is geographic — so the gap is
    always a true distance and never degrees. Event geometries are the union of
    the member patches in the input CRS, not the buffered hulls.

    Parameters
    ----------
    patches : geopandas.GeoDataFrame
        Patches as produced by the ``patches_*`` functions: ``patch_id``,
        ``pixel_count``, ``area_ha`` and geometry.
    gap_m : float
        Maximum separation in metres between two patches of the same event.
    group_by : sequence of str, optional
        Columns that must match for two patches to merge, for example
        ``["time"]`` to keep time steps apart. Without it, all patches are
        candidates.
    metric_crs : optional
        CRS to buffer in, overriding the estimated UTM zone. Useful when
        patches straddle more than one UTM zone.

    Returns
    -------
    events : geopandas.GeoDataFrame
        ``event_id``, ``area_ha`` (sum of member areas), ``patch_count``,
        ``pixel_count``, the ``group_by`` columns, and the dissolved member
        geometry, in the input CRS. Member areas are summed, not remeasured, so
        patches that overlap in space (for example the same location in several
        time steps merged without ``group_by=["time"]``) count once per patch.
        Pass ``group_by`` to keep such patches apart, or measure the dissolved
        geometry when a footprint area is wanted.
    membership : pandas.DataFrame
        Two columns, ``event_id`` and ``patch_id``, one row per member patch.

    Raises
    ------
    ValueError
        If ``gap_m`` is negative or ``patches`` has no CRS.
    """
    gpd = _require_geopandas()

    if gap_m < 0:
        raise ValueError(f"gap_m must be non-negative, got {gap_m}")

    group_cols = list(group_by or [])
    event_columns = ["event_id", *group_cols, "area_ha", "patch_count", "pixel_count"]

    if len(patches) == 0:
        empty = gpd.GeoDataFrame(
            {col: [] for col in event_columns}, geometry=[], crs=patches.crs
        )
        return empty, pd.DataFrame({"event_id": [], "patch_id": []})

    work_crs = _metric_crs(patches, metric_crs)
    projected = patches.to_crs(work_crs)
    buffered = projected.geometry.buffer(gap_m / 2.0).to_numpy()

    if group_cols:
        groups = patches.groupby(group_cols, sort=False, dropna=False).indices.values()
    else:
        groups = [np.arange(len(patches))]

    event_rows: list[dict[str, Any]] = []
    event_geoms = []
    membership_rows: list[dict[str, Any]] = []
    next_event_id = 1

    source_geoms = patches.geometry.to_numpy()
    for positions in groups:
        positions = np.asarray(positions)
        labels = _connected_components(buffered[positions], len(positions))
        for label in np.unique(labels):
            members = positions[labels == label]
            rows = patches.iloc[members]
            row: dict[str, Any] = {"event_id": next_event_id}
            for col in group_cols:
                row[col] = rows[col].iloc[0]
            row["area_ha"] = float(rows["area_ha"].sum())
            row["patch_count"] = int(len(members))
            row["pixel_count"] = int(rows["pixel_count"].sum())
            event_rows.append(row)
            event_geoms.append(shapely.union_all(source_geoms[members]))
            for patch_id in rows["patch_id"]:
                membership_rows.append(
                    {"event_id": next_event_id, "patch_id": int(patch_id)}
                )
            next_event_id += 1

    events = gpd.GeoDataFrame(
        pd.DataFrame(event_rows, columns=event_columns),
        geometry=event_geoms,
        crs=patches.crs,
    )
    membership = pd.DataFrame(membership_rows, columns=["event_id", "patch_id"])
    return events, membership


def assign_to_polygons(
    events: "GeoDataFrame",
    polygons: "GeoDataFrame",
    *,
    polygon_id: str,
    min_overlap_frac: float = 0.0,
) -> pd.DataFrame:
    """Relate events to reference polygons, many-to-many, by area of overlap.

    Both inputs are measured in an equal-area projection, so the fractions are
    comparable regardless of the CRS they arrive in. An event overlapping
    several polygons produces one row per polygon; pairs that do not overlap are
    omitted.

    Parameters
    ----------
    events : geopandas.GeoDataFrame
        Events carrying an ``event_id`` column and geometry.
    polygons : geopandas.GeoDataFrame
        Reference polygons carrying ``polygon_id`` and geometry.
    polygon_id : str
        Name of the identifying column in ``polygons``.
    min_overlap_frac : float, default 0.0
        Drop pairs whose ``overlap_frac`` falls below this value.

    Returns
    -------
    pandas.DataFrame
        Columns ``event_id``, ``<polygon_id>``, ``overlap_area_ha`` and
        ``overlap_frac`` (the share of the event's area inside that polygon).

    Raises
    ------
    ValueError
        If ``polygon_id`` is missing from ``polygons``, ``event_id`` is missing
        from ``events``, or either input has no CRS.
    """
    gpd = _require_geopandas()

    if "event_id" not in events.columns:
        raise ValueError("events must carry an 'event_id' column.")
    if polygon_id not in polygons.columns:
        raise ValueError(f"polygons has no '{polygon_id}' column.")
    if events.crs is None or polygons.crs is None:
        raise ValueError("both events and polygons must have a CRS set.")

    out_columns = ["event_id", polygon_id, "overlap_area_ha", "overlap_frac"]
    if len(events) == 0 or len(polygons) == 0:
        return pd.DataFrame({col: [] for col in out_columns})

    equal_area = pyproj.CRS.from_epsg(EQUAL_AREA_EPSG)
    ev = (
        events[["event_id", events.geometry.name]]
        .set_geometry(events.geometry.name)
        .to_crs(equal_area)
        .reset_index(drop=True)
    )
    poly = (
        polygons[[polygon_id, polygons.geometry.name]]
        .set_geometry(polygons.geometry.name)
        .to_crs(equal_area)
        .reset_index(drop=True)
    )

    joined = gpd.sjoin(ev, poly, predicate="intersects", how="inner")
    if len(joined) == 0:
        return pd.DataFrame({col: [] for col in out_columns})

    left_positions = ev.index.get_indexer(joined.index)
    right_positions = joined["index_right"].to_numpy()
    intersections = shapely.intersection(
        ev.geometry.to_numpy()[left_positions],
        poly.geometry.to_numpy()[right_positions],
    )
    overlap_ha = shapely.area(intersections) * M2_TO_HA
    event_ha = shapely.area(ev.geometry.to_numpy()[left_positions]) * M2_TO_HA

    result = pd.DataFrame(
        {
            "event_id": joined["event_id"].to_numpy(),
            polygon_id: joined[polygon_id].to_numpy(),
            "overlap_area_ha": overlap_ha,
            "overlap_frac": np.divide(
                overlap_ha,
                event_ha,
                out=np.zeros_like(overlap_ha),
                where=event_ha > 0,
            ),
        }
    )
    result = result[result["overlap_area_ha"] > 0]
    if min_overlap_frac > 0:
        result = result[result["overlap_frac"] >= min_overlap_frac]
    return result.reset_index(drop=True)


def write_geoparquet(gdf: "GeoDataFrame", path) -> None:
    """Write a GeoDataFrame to GeoParquet 1.0.0 with its CRS recorded.

    The schema version is pinned to the stable 1.0.0 specification and geometry
    is WKB-encoded, the combination readable by the widest set of tools.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Table to write. It must have a CRS.
    path : str or path-like
        Destination file.

    Raises
    ------
    ValueError
        If ``gdf`` has no CRS, which would make readers fall back to OGC:CRS84.
    """
    _require_geopandas()
    _require_pyarrow()
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS. GeoParquet readers default a missing CRS "
            "to OGC:CRS84; set the CRS before writing."
        )
    gdf.to_parquet(
        path,
        index=False,
        schema_version="1.0.0",
        geometry_encoding="WKB",
    )


def read_geoparquet(path) -> "GeoDataFrame":
    """Read a GeoParquet file written by :func:`write_geoparquet`.

    Parameters
    ----------
    path : str or path-like
        Source file.

    Returns
    -------
    geopandas.GeoDataFrame
        The table with its CRS restored from the GeoParquet metadata.
    """
    gpd = _require_geopandas()
    _require_pyarrow()
    return gpd.read_parquet(path)


__all__ = [
    "patches_from_mask",
    "patches_from_categorical",
    "patches_over_time",
    "merge_patches",
    "assign_to_polygons",
    "write_geoparquet",
    "read_geoparquet",
]

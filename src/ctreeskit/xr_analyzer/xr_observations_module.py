"""Normalize detection layers from different sources into one observation record.

An *observation* is a detection episode at a single pixel (or point) from a
single source: where it is, which source and sensor saw it, when it was first
seen, confirmed and last seen, how many looks the source had and how many were
positive, what state it is in, how confident it is, an optional class hint, how
precise the dates are, and which snapshot it was read from. Pixels that never
detected anything are not observations -- they contribute only the look counts
carried on the detections around them.

Four adapters bring common source layouts into that record:

* :func:`observations_from_dated_codes` -- a single raster whose pixel values
  are *dated state codes* (a state band plus an encoded year/month).
* :func:`observations_from_annual_alert_days` -- annual slices with a per-pixel
  day value giving the date of an alert within each year.
* :func:`observations_from_tier_steps` -- a regularly stepped categorical cube
  whose pixels carry a *tier code* at one step only, the step supplying the
  date.
* :func:`observations_from_points` -- a table of point detections with a
  timestamp and an optional confidence.

The raster adapters return an :class:`xarray.Dataset` on the **source grid** --
nothing is resampled, so a normalized record is always directly comparable with
the pixels it came from. :func:`observations_to_table` flattens any of them to
one flat row per detected pixel, and every adapter yields the same column set
and dtypes, so records from different sources concatenate.
:func:`observations_mask` turns a record back into a boolean raster mask, which
is the bridge to :mod:`~ctreeskit.xr_analyzer.xr_vectorize_module` --
``patches_from_mask`` takes it directly.

Parquet output depends on pyarrow, which ships in the optional ``vector``
extra: ``pip install 'ctreeskit[vector]'``. The import is deferred, so importing
this module without the extra succeeds and only the functions that need it
raise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyproj
import xarray as xr

from .xr_vectorize_module import _require_pyarrow

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

#: State codes carried in the ``state`` variable.
#:
#: =====  ================  ====================================================
#: Code   Name              Meaning
#: =====  ================  ====================================================
#: 0      none              No detection at this pixel.
#: 1      candidate         Detected once, not yet corroborated by the source.
#: 2      confirmed         Corroborated by the source's own confirmation rule.
#: 3      rejected          Detected then withdrawn by the source.
#: 4      alert             A single-look near-real-time detection.
#: 5      point_detection   A point observation rasterized or kept as a point.
#: 255    fill              No data; the source had nothing to say here.
#: =====  ================  ====================================================
STATE_CODES: dict[str, int] = {
    "none": 0,
    "candidate": 1,
    "confirmed": 2,
    "rejected": 3,
    "alert": 4,
    "point_detection": 5,
}

#: Value marking "no data" in ``state``. Distinct from ``0`` (no detection).
STATE_FILL = 255

#: Date-precision codes carried in ``date_precision``.
#:
#: ====  ============  ==========================================================
#: Code  Name          Meaning
#: ====  ============  ==========================================================
#: 0     unknown       No date.
#: 1     day           The date is the observed day.
#: 2     month         The date is the first of the month the detection falls in.
#: 3     lower_bound   The detection is no earlier than the date, and may be
#:                     considerably later (the source could not observe in the
#:                     interval before it).
#: ====  ============  ==========================================================
DATE_PRECISION: dict[str, int] = {
    "unknown": 0,
    "day": 1,
    "month": 2,
    "lower_bound": 3,
}


@dataclass(frozen=True)
class ObservationField:
    """One variable of the observation record.

    Attributes
    ----------
    name : str
        Variable name in the normalized Dataset and column name in the table.
    dtype : str
        NumPy dtype the variable always carries.
    fill : object
        The value meaning "not known" for this field.
    description : str
        What the field holds.
    """

    name: str
    dtype: str
    fill: Any
    description: str


#: The normalized variables, in order, with their dtypes and "unknown" values.
OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField(
        "state", "uint8", STATE_FILL, "Detection state; see STATE_CODES."
    ),
    ObservationField(
        "first_obs_date",
        "datetime64[ns]",
        "NaT",
        "Date of the first observation of this episode.",
    ),
    ObservationField(
        "confirm_date",
        "datetime64[ns]",
        "NaT",
        "Date the source considered the detection confirmed; NaT if it never was.",
    ),
    ObservationField(
        "last_obs_date",
        "datetime64[ns]",
        "NaT",
        "Date of the last observation of this episode.",
    ),
    ObservationField(
        "n_obs", "uint16", 0, "Number of looks the source had at this pixel."
    ),
    ObservationField(
        "n_positive", "uint16", 0, "How many of those looks were positive."
    ),
    ObservationField(
        "confidence_native",
        "float32",
        "NaN",
        "Confidence as the source expresses it, on its own scale.",
    ),
    ObservationField(
        "confidence", "float32", "NaN", "Confidence rescaled to [0, 1]."
    ),
    ObservationField(
        "class_hint",
        "uint8",
        0,
        "Optional source-supplied class or driver hint; 0 means none.",
    ),
    ObservationField(
        "date_precision", "uint8", 0, "Date precision; see DATE_PRECISION."
    ),
)

#: Dataset attributes every adapter sets alongside the CRS written by rioxarray.
OBSERVATION_ATTRS: tuple[str, ...] = (
    "source",
    "sensor",
    "pixel_size_m",
    "snapshot_id",
    "read_at",
    "has_counts",
)

#: Columns of the flat table, in order, with the dtype each always carries.
#: ``n_obs`` and ``n_positive`` are 0 unless the dataset's ``has_counts``
#: attribute is true, in which case they are real counts.
OBSERVATION_TABLE_DTYPES: dict[str, str] = {
    "x": "float64",
    "y": "float64",
    **{field.name: field.dtype for field in OBSERVATION_FIELDS},
    "source": "category",
    "sensor": "category",
    "pixel_size_m": "float32",
    "snapshot_id": "category",
}

_NAT_INT = np.iinfo(np.int64).min
_NAT = np.datetime64("NaT", "ns")

#: Default code bands for :func:`observations_from_dated_codes`: the confirmed
#: band starts at 0, candidates at 10000, rejections at 20000.
DEFAULT_CODE_OFFSETS: dict[str, int] = {
    "confirmed": 0,
    "candidate": 10000,
    "rejected": 20000,
}

_EPOCH_BAND_WIDTH = {"YYMM": 10_000, "YYYYMM": 1_000_000}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _spatial_dims(obj: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    """Return the (y, x) dimension names rioxarray reports for a raster."""
    try:
        return obj.rio.y_dim, obj.rio.x_dim
    except Exception:
        if "y" in obj.dims and "x" in obj.dims:
            return "y", "x"
        raise ValueError(
            "Could not determine the spatial dimensions. Set them with "
            "`obj.rio.set_spatial_dims(x_dim=..., y_dim=...)`."
        )


def _infer_pixel_size_m(da: xr.DataArray) -> float:
    """Pixel width in metres, or NaN when the grid is not metre-based."""
    crs = da.rio.crs
    if crs is None:
        return float("nan")
    crs = pyproj.CRS.from_user_input(crs)
    if crs.is_geographic:
        return float("nan")
    units = {axis.unit_name for axis in crs.axis_info}
    if not units <= {"metre", "meter", "m"}:
        return float("nan")
    return float(abs(da.rio.transform().a))


def _now_iso() -> str:
    """Current UTC time as an ISO 8601 string, for the ``read_at`` attribute."""
    return datetime.now(timezone.utc).isoformat()


def _record_attrs(
    *,
    source: str,
    sensor: str,
    pixel_size_m: float,
    snapshot_id: str | None,
    has_counts: bool,
) -> dict[str, Any]:
    """Assemble the provenance attributes carried on every normalized record."""
    return {
        "source": source,
        "sensor": sensor,
        "pixel_size_m": float(pixel_size_m),
        "snapshot_id": "" if snapshot_id is None else str(snapshot_id),
        "read_at": _now_iso(),
        "has_counts": has_counts,
    }


def _months_to_datetime64(months: np.ndarray) -> np.ndarray:
    """Months since 1970-01 (``_NAT_INT`` for unknown) to datetime64[ns]."""
    return months.astype("datetime64[M]").astype("datetime64[ns]")


def _days_to_datetime64(days: np.ndarray) -> np.ndarray:
    """Days since 1970-01-01 (``_NAT_INT`` for unknown) to datetime64[ns]."""
    return days.astype("datetime64[D]").astype("datetime64[ns]")


def _constant(like: xr.DataArray, value: Any, dtype: str) -> xr.DataArray:
    """A constant array shaped and chunked like ``like``, staying lazy."""
    return xr.full_like(like, value, dtype=dtype)


def _nat_like(like: xr.DataArray) -> xr.DataArray:
    """An all-NaT datetime64[ns] array shaped and chunked like ``like``."""
    return xr.full_like(like, _NAT_INT, dtype="int64").astype("datetime64[ns]")


def _assemble(
    variables: dict[str, xr.DataArray],
    *,
    crs: Any,
    attrs: dict[str, Any],
) -> xr.Dataset:
    """Build the normalized Dataset from its variables and write the CRS."""
    ds = xr.Dataset(variables, attrs=attrs)
    if crs is not None:
        ds = ds.rio.write_crs(crs)
    return ds


# ---------------------------------------------------------------------------
# Adapter 1: dated state codes
# ---------------------------------------------------------------------------


def _decode_dated_codes_block(
    codes: np.ndarray,
    *,
    band_width: int,
    band_to_state: dict[int, int],
    confirmed_state: int | None,
    century: int,
    add_century: bool,
    stable: tuple[int, ...],
    fill_value: int | None,
    base_precision: int,
    lower_bound_months: tuple[int, ...],
    class_by_band: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode one chunk of dated state codes. Pure NumPy, no cross-chunk reads."""
    codes = np.asarray(codes).astype("int64", copy=False)
    valid = codes >= 0
    for value in stable:
        valid &= codes != value
    if fill_value is not None:
        valid &= codes != fill_value

    band = np.where(valid, codes // band_width, -1)
    encoded = np.where(valid, codes % band_width, 0)
    year = encoded // 100
    month = encoded % 100
    valid &= (month >= 1) & (month <= 12)

    state = np.zeros(codes.shape, dtype="uint8")
    class_hint = np.zeros(codes.shape, dtype="uint8")
    known = np.zeros(codes.shape, dtype=bool)
    for band_index, state_code in band_to_state.items():
        selected = valid & (band == band_index)
        state = np.where(selected, state_code, state).astype("uint8")
        hint = class_by_band.get(band_index, 0)
        if hint:
            class_hint = np.where(selected, hint, class_hint).astype("uint8")
        known |= selected
    valid &= known
    state = np.where(valid, state, 0).astype("uint8")
    class_hint = np.where(valid, class_hint, 0).astype("uint8")

    full_year = year + century if add_century else year
    months = np.where(valid, (full_year - 1970) * 12 + (month - 1), _NAT_INT)
    first = _months_to_datetime64(months.astype("int64"))

    if confirmed_state is None:
        confirm = np.full(codes.shape, _NAT, dtype="datetime64[ns]")
    else:
        confirm = np.where(state == confirmed_state, first, _NAT)

    precision = np.where(valid, base_precision, 0).astype("uint8")
    if lower_bound_months:
        in_gap = np.isin(month, np.asarray(lower_bound_months, dtype="int64"))
        precision = np.where(
            valid & in_gap, DATE_PRECISION["lower_bound"], precision
        ).astype("uint8")

    return state, first, confirm, precision, class_hint


def observations_from_dated_codes(
    da: xr.DataArray,
    *,
    code_table: Mapping[str, int] | None = None,
    epoch_format: str = "YYMM",
    century: int = 2000,
    offsets: Mapping[str, int] = DEFAULT_CODE_OFFSETS,
    stable_values: Sequence[int] = (0,),
    fill_value: int | None = None,
    date_precision: str = "month",
    lower_bound_months: Sequence[int] | None = None,
    class_hint_map: Mapping[str, int] | None = None,
    snapshot_id: str | None = None,
    source: str = "unknown",
    sensor: str = "unknown",
    pixel_size_m: float | None = None,
) -> xr.Dataset:
    """Normalize a raster of dated state codes into an observation record.

    The input is a single 2-D integer raster with **no time dimension**: each
    pixel holds one code combining a state band with an encoded date, for
    example ``2403`` for a confirmed detection in March 2024, ``12403`` for a
    candidate in the same month and ``22403`` for a rejected one. Pixels holding
    a stable or fill value carry no detection and decode to ``state = 0``.

    The output is 2-D on the source grid: one detection episode per pixel, so no
    time dimension is needed. Nothing is resampled.

    Parameters
    ----------
    da : xr.DataArray
        2-D integer raster with a rioxarray CRS. May be dask-backed; the decode
        is applied chunk by chunk and stays lazy.
    code_table : mapping of str to int, optional
        State name to output state code. Defaults to :data:`STATE_CODES`, so a
        band named ``"confirmed"`` decodes to state 2. Supply it to route a
        source's own vocabulary onto different codes.
    epoch_format : {"YYMM", "YYYYMM"}, default "YYMM"
        How the date is encoded inside the band: a two-digit or four-digit year
        followed by a two-digit month.
    century : int, default 2000
        Added to a two-digit year under ``"YYMM"``. Ignored for ``"YYYYMM"``.
    offsets : mapping of str to int, default :data:`DEFAULT_CODE_OFFSETS`
        State name to the additive offset that marks its band. Each offset must
        be a whole multiple of the band width implied by ``epoch_format``
        (10000 for ``"YYMM"``, 1000000 for ``"YYYYMM"``).
    stable_values : sequence of int, default (0,)
        Values meaning "observed, nothing detected". They decode to state 0.
    fill_value : int, optional
        Value meaning "no data". It also decodes to state 0.
    date_precision : {"day", "month", "lower_bound"}, default "month"
        Precision to record for every decoded date. A ``YYMM``-style code names
        a month, not a day, so the default is ``"month"`` and the date is the
        first of that month.
    lower_bound_months : sequence of int, optional
        Month numbers (1-12) whose detections are recorded as
        ``date_precision = 3`` (lower bound) instead. Use it for a source that
        cannot observe during part of the year and dates everything it missed to
        the first month it could see again: those dates bound the detection from
        below rather than locating it.
    class_hint_map : mapping of str to int, optional
        State name to a class hint value written into ``class_hint``.
    snapshot_id : str, optional
        Identifier of the snapshot the raster was read from.
    source, sensor : str
        Free-text provenance recorded on the dataset.
    pixel_size_m : float, optional
        Pixel size in metres. Inferred from the transform on a metre-based
        projected grid; NaN when it cannot be inferred.

    Returns
    -------
    xr.Dataset
        The normalized record: the variables of :data:`OBSERVATION_FIELDS` on
        the input's ``(y, x)`` grid, with the CRS and the attributes of
        :data:`OBSERVATION_ATTRS`. ``last_obs_date`` is NaT -- a single dated
        code says when a detection happened, not when it was last seen -- and
        ``has_counts`` is False.

    Raises
    ------
    ValueError
        If ``da`` is not 2-D, ``epoch_format`` is unknown, ``date_precision`` is
        unknown, or an offset is not a whole multiple of the band width.
    """
    if da.ndim != 2:
        raise ValueError(
            f"da must be 2-D; got dims {da.dims}. Dated state codes carry their "
            "own date, so the layer has no time dimension."
        )
    if epoch_format not in _EPOCH_BAND_WIDTH:
        raise ValueError(
            f"epoch_format must be one of {sorted(_EPOCH_BAND_WIDTH)}, "
            f"got {epoch_format!r}"
        )
    if date_precision not in DATE_PRECISION:
        raise ValueError(
            f"date_precision must be one of {sorted(DATE_PRECISION)}, "
            f"got {date_precision!r}"
        )

    band_width = _EPOCH_BAND_WIDTH[epoch_format]
    table = dict(code_table) if code_table is not None else dict(STATE_CODES)
    hints = dict(class_hint_map or {})

    band_to_state: dict[int, int] = {}
    class_by_band: dict[int, int] = {}
    for name, offset in offsets.items():
        if offset % band_width:
            raise ValueError(
                f"offset {offset} for state {name!r} is not a whole multiple of "
                f"the {epoch_format} band width ({band_width})."
            )
        if name not in table:
            raise ValueError(
                f"state {name!r} has an offset but no entry in code_table "
                f"({sorted(table)})."
            )
        band_to_state[offset // band_width] = int(table[name])
        if name in hints:
            class_by_band[offset // band_width] = int(hints[name])

    confirmed_state = table.get("confirmed")

    state, first, confirm, precision, class_hint = xr.apply_ufunc(
        _decode_dated_codes_block,
        da,
        kwargs={
            "band_width": band_width,
            "band_to_state": band_to_state,
            "confirmed_state": confirmed_state,
            "century": century,
            "add_century": epoch_format == "YYMM",
            "stable": tuple(int(v) for v in stable_values),
            "fill_value": None if fill_value is None else int(fill_value),
            "base_precision": DATE_PRECISION[date_precision],
            "lower_bound_months": tuple(int(m) for m in (lower_bound_months or ())),
            "class_by_band": class_by_band,
        },
        output_core_dims=[[], [], [], [], []],
        dask="parallelized",
        output_dtypes=["uint8", "datetime64[ns]", "datetime64[ns]", "uint8", "uint8"],
    )

    if pixel_size_m is None:
        pixel_size_m = _infer_pixel_size_m(da)

    variables = {
        "state": state,
        "first_obs_date": first,
        "confirm_date": confirm,
        "last_obs_date": _nat_like(state),
        "n_obs": _constant(state, 0, "uint16"),
        "n_positive": _constant(state, 0, "uint16"),
        "confidence_native": _constant(state, np.nan, "float32"),
        "confidence": _constant(state, np.nan, "float32"),
        "class_hint": class_hint,
        "date_precision": precision,
    }
    return _assemble(
        variables,
        crs=da.rio.crs,
        attrs=_record_attrs(
            source=source,
            sensor=sensor,
            pixel_size_m=pixel_size_m,
            snapshot_id=snapshot_id,
            has_counts=False,
        ),
    )


# ---------------------------------------------------------------------------
# Adapter 2: annual slices with a per-pixel alert day
# ---------------------------------------------------------------------------


def observations_from_annual_alert_days(
    alert: xr.DataArray,
    date: xr.DataArray,
    *,
    day_epoch: str = "1970-01-01",
    time_dim: str = "time",
    fill_value: int = -9999,
    alert_fill_value: int | None = None,
    class_hint_map: Mapping[int, int] | None = None,
    snapshot_id: str | None = None,
    source: str = "unknown",
    sensor: str = "unknown",
    pixel_size_m: float | None = None,
) -> xr.Dataset:
    """Normalize annual alert slices with a per-pixel day value.

    The input is a pair of rasters with an annual ``time`` dimension: ``alert``,
    where a non-zero value marks an alert, and ``date``, holding the day of that
    alert as an integer offset from ``day_epoch``.

    The ``time`` dimension is **kept** in the output rather than collapsed. A
    pixel may alert in more than one year, and those are separate detection
    episodes -- collapsing them would silently drop all but one. Each annual
    layer is therefore its own set of episodes, and the dimension is what keeps
    them apart. (Where a source's episodes cannot overlap, as with
    :func:`observations_from_dated_codes`, the record is plain 2-D instead.)

    Parameters
    ----------
    alert : xr.DataArray
        ``(time, y, x)`` raster; non-zero marks an alert. Its non-zero values
        may also encode a class, which ``class_hint_map`` can carry through.
        A band that marks its own nodata with a non-zero value needs
        ``alert_fill_value``, or that value reads as an alert.
    date : xr.DataArray
        ``(time, y, x)`` raster of integer day offsets from ``day_epoch``,
        aligned with ``alert``.
    day_epoch : str, default "1970-01-01"
        Date the day offsets count from.
    time_dim : str, default "time"
        Name of the annual dimension.
    fill_value : int, default -9999
        Value in ``date`` meaning "no date". Such pixels decode to state 0.
    alert_fill_value : int, optional
        Value in ``alert`` meaning "no data". Such pixels decode to state 0
        however ``date`` reads there. Set it whenever the alert band's nodata
        is not 0 -- a band filled with a negative sentinel is non-zero, so
        without this every filled pixel carrying a readable date would decode
        to an alert.
    class_hint_map : mapping of int to int, optional
        Alert value to a class hint value written into ``class_hint``.
    snapshot_id : str, optional
        Identifier of the snapshot the rasters were read from.
    source, sensor : str
        Free-text provenance recorded on the dataset.
    pixel_size_m : float, optional
        Pixel size in metres; inferred from the transform when omitted.

    Returns
    -------
    xr.Dataset
        The normalized record on the ``(time, y, x)`` source grid.
        ``first_obs_date``, ``confirm_date`` and ``last_obs_date`` all hold the
        decoded day, ``date_precision`` is 1 (day) and ``state`` is 4 (alert)
        wherever an alert was decoded. ``has_counts`` is False.

    Raises
    ------
    ValueError
        If ``time_dim`` is missing from either input or their shapes disagree.
    """
    if time_dim not in alert.dims:
        raise ValueError(f"'{time_dim}' is not a dimension of alert ({alert.dims}).")
    if time_dim not in date.dims:
        raise ValueError(f"'{time_dim}' is not a dimension of date ({date.dims}).")
    if alert.shape != date.shape:
        raise ValueError(
            f"alert and date must share a shape; got {alert.shape} and {date.shape}."
        )

    epoch_offset = int(
        (np.datetime64(day_epoch, "D") - np.datetime64("1970-01-01", "D")).astype(
            "int64"
        )
    )

    detected = (alert != 0) & (date != fill_value)
    if alert_fill_value is not None:
        detected = detected & (alert != alert_fill_value)
    state = xr.where(detected, STATE_CODES["alert"], 0).astype("uint8")

    days = xr.where(detected, date.astype("int64") + epoch_offset, _NAT_INT)
    day_date = xr.apply_ufunc(
        _days_to_datetime64,
        days.astype("int64"),
        dask="parallelized",
        output_dtypes=["datetime64[ns]"],
    )

    class_hint = _constant(state, 0, "uint8")
    for alert_value, hint in (class_hint_map or {}).items():
        class_hint = xr.where(
            detected & (alert == alert_value), int(hint), class_hint
        ).astype("uint8")

    if pixel_size_m is None:
        pixel_size_m = _infer_pixel_size_m(alert)

    variables = {
        "state": state,
        "first_obs_date": day_date,
        "confirm_date": day_date,
        "last_obs_date": day_date,
        "n_obs": _constant(state, 0, "uint16"),
        "n_positive": _constant(state, 0, "uint16"),
        "confidence_native": _constant(state, np.nan, "float32"),
        "confidence": _constant(state, np.nan, "float32"),
        "class_hint": class_hint,
        "date_precision": xr.where(detected, DATE_PRECISION["day"], 0).astype("uint8"),
    }
    return _assemble(
        variables,
        crs=alert.rio.crs,
        attrs=_record_attrs(
            source=source,
            sensor=sensor,
            pixel_size_m=pixel_size_m,
            snapshot_id=snapshot_id,
            has_counts=False,
        ),
    )


def select_month(ds: xr.Dataset, year: int, month: int) -> xr.DataArray:
    """Boolean mask of the alerts in ``ds`` whose day falls in a given month.

    Parameters
    ----------
    ds : xr.Dataset
        A normalized record, typically from
        :func:`observations_from_annual_alert_days`.
    year, month : int
        Calendar year and month number (1-12) to select.

    Returns
    -------
    xr.DataArray
        A boolean array shaped like ``ds.state``, True where an alert's
        ``first_obs_date`` falls in that month.
    """
    dates = ds["first_obs_date"]
    return (
        (ds["state"] == STATE_CODES["alert"])
        & (dates.dt.year == year)
        & (dates.dt.month == month)
    )


# ---------------------------------------------------------------------------
# Adapter 3: point detections
# ---------------------------------------------------------------------------


def _normalize_confidence(
    values: pd.Series,
    confidence_map: Mapping[Any, float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (native, normalized) confidence arrays from a source column.

    With ``confidence_map`` the column is categorical: the mapped value is the
    normalized confidence and the native column has no numeric reading, so
    ``confidence_native`` is NaN. Without it the column must be numeric; values
    above 1 are read as percentages and divided by 100, and the raw value is
    kept as ``confidence_native``.
    """
    if confidence_map is not None:
        mapped = values.map(confidence_map).astype("float64").to_numpy()
        native = np.full(len(values), np.nan, dtype="float64")
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
        if np.isfinite(numeric).any():
            native = numeric
        return native, mapped

    native = pd.to_numeric(values, errors="coerce").to_numpy(dtype="float64")
    finite = native[np.isfinite(native)]
    scale = 100.0 if finite.size and finite.max() > 1.0 else 1.0
    return native, native / scale


def observations_from_points(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    datetime_col: str,
    confidence_col: str | None = None,
    confidence_map: Mapping[Any, float] | None = None,
    sensor_col: str | None = None,
    class_col: str | None = None,
    class_hint_map: Mapping[Any, int] | None = None,
    crs: Any,
    source: str = "unknown",
    sensor: str = "unknown",
    snapshot_id: str | None = None,
) -> pd.DataFrame:
    """Normalize a table of point detections into the flat observation table.

    Point detections have no grid, so this adapter produces the table form
    directly rather than a Dataset. Use :func:`points_to_grid_mask` to place the
    points on a reference grid for co-location with raster records.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per point detection.
    x_col, y_col : str
        Columns holding the coordinates, in ``crs``.
    datetime_col : str
        Column holding the detection timestamp; parsed with
        :func:`pandas.to_datetime`.
    confidence_col : str, optional
        Column holding the source's confidence.
    confidence_map : mapping, optional
        Maps a categorical confidence (for example ``{"low": 0.2, "nominal":
        0.6, "high": 0.9}``) onto ``confidence`` in [0, 1]. Without it the
        column must be numeric; values above 1 are read as percentages.
    sensor_col : str, optional
        Column naming the sensor per row, overriding ``sensor``.
    class_col : str, optional
        Column holding a source class, mapped through ``class_hint_map``.
    class_hint_map : mapping, optional
        Maps ``class_col`` values onto ``class_hint``.
    crs : any
        CRS the coordinates are in, as anything
        :meth:`pyproj.CRS.from_user_input` accepts. Recorded on the returned
        table's ``attrs["crs"]``.
    source, sensor : str
        Free-text provenance written onto every row.
    snapshot_id : str, optional
        Identifier of the snapshot the points were read from.

    Returns
    -------
    pandas.DataFrame
        The flat table described by :data:`OBSERVATION_TABLE_DTYPES`, with
        ``state = 5`` (point detection) and ``date_precision = 1`` (day). The
        dates are all the detection timestamp; ``pixel_size_m`` is NaN, since a
        point covers no cell.

    Raises
    ------
    ValueError
        If a named column is missing from ``df``.
    """
    required = [x_col, y_col, datetime_col]
    for name in (confidence_col, sensor_col, class_col):
        if name is not None:
            required.append(name)
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"df has no column(s) {missing}; got {list(df.columns)}.")

    n = len(df)
    when = pd.to_datetime(df[datetime_col]).to_numpy(dtype="datetime64[ns]")

    if confidence_col is None:
        native = np.full(n, np.nan, dtype="float64")
        normalized = np.full(n, np.nan, dtype="float64")
    else:
        native, normalized = _normalize_confidence(df[confidence_col], confidence_map)

    if class_col is None:
        class_hint = np.zeros(n, dtype="int64")
    else:
        class_hint = (
            df[class_col].map(class_hint_map or {}).fillna(0).astype("int64").to_numpy()
        )

    table = pd.DataFrame(
        {
            "x": df[x_col].to_numpy(dtype="float64"),
            "y": df[y_col].to_numpy(dtype="float64"),
            "state": np.full(n, STATE_CODES["point_detection"], dtype="int64"),
            "first_obs_date": when,
            "confirm_date": when,
            "last_obs_date": when,
            "n_obs": np.zeros(n, dtype="int64"),
            "n_positive": np.zeros(n, dtype="int64"),
            "confidence_native": native,
            "confidence": normalized,
            "class_hint": class_hint,
            "date_precision": np.full(n, DATE_PRECISION["day"], dtype="int64"),
            "source": np.full(n, source, dtype=object),
            "sensor": (
                df[sensor_col].astype(object).to_numpy()
                if sensor_col is not None
                else np.full(n, sensor, dtype=object)
            ),
            "pixel_size_m": np.full(n, np.nan, dtype="float64"),
            "snapshot_id": np.full(
                n, "" if snapshot_id is None else str(snapshot_id), dtype=object
            ),
        }
    )
    table = _finalize_table(table)
    table.attrs["crs"] = pyproj.CRS.from_user_input(crs).to_wkt()
    return table


def points_to_grid_mask(
    table: pd.DataFrame,
    like: xr.DataArray,
    *,
    crs: Any = None,
) -> xr.DataArray:
    """Rasterize a point observation table onto a reference grid as a mask.

    Points are reprojected to the reference grid's CRS when they differ, then
    each is assigned to the cell containing it. Points falling outside the grid
    are dropped.

    Parameters
    ----------
    table : pandas.DataFrame
        A point table from :func:`observations_from_points`, carrying ``x`` and
        ``y`` columns.
    like : xr.DataArray
        Reference grid supplying the shape, coordinates, CRS and transform.
    crs : any, optional
        CRS of the table's coordinates, overriding ``table.attrs["crs"]``.
        Required when the table has lost its attributes, as happens across a
        Parquet round trip.

    Returns
    -------
    xr.DataArray
        A boolean ``(y, x)`` array on ``like``'s grid, True in cells holding at
        least one point, carrying ``like``'s CRS.

    Raises
    ------
    ValueError
        If ``like`` has no CRS, or the table's CRS is neither given nor recorded.
    """
    target_crs = like.rio.crs
    if target_crs is None:
        raise ValueError(
            "like has no CRS. Set one with `like.rio.write_crs(...)` first."
        )
    source_crs = crs if crs is not None else table.attrs.get("crs")
    if source_crs is None:
        raise ValueError(
            "The point table carries no CRS. Pass crs=... (attributes do not "
            "survive a Parquet round trip)."
        )

    y_dim, x_dim = _spatial_dims(like)
    xs = table["x"].to_numpy(dtype="float64")
    ys = table["y"].to_numpy(dtype="float64")

    source_crs = pyproj.CRS.from_user_input(source_crs)
    target = pyproj.CRS.from_user_input(target_crs)
    if not source_crs.equals(target):
        transformer = pyproj.Transformer.from_crs(source_crs, target, always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    inverse = ~like.rio.transform()
    cols, rows = inverse * (xs, ys)
    cols = np.floor(np.asarray(cols)).astype("int64")
    rows = np.floor(np.asarray(rows)).astype("int64")

    ny = like.sizes[y_dim]
    nx = like.sizes[x_dim]
    inside = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)

    mask = np.zeros((ny, nx), dtype=bool)
    mask[rows[inside], cols[inside]] = True

    out = xr.DataArray(
        mask,
        coords={y_dim: like[y_dim], x_dim: like[x_dim]},
        dims=(y_dim, x_dim),
        name="point_mask",
    )
    return out.rio.write_crs(target_crs)


# ---------------------------------------------------------------------------
# Adapter 4: one-step tier codes
# ---------------------------------------------------------------------------


def _normalize_tier_map(tier_map: Mapping[int, str | int]) -> dict[int, int]:
    """Resolve a source tier vocabulary onto :data:`STATE_CODES` values."""
    if not tier_map:
        raise ValueError("tier_map is empty; it must map at least one source value.")
    known = set(STATE_CODES.values())
    resolved: dict[int, int] = {}
    for value, state in tier_map.items():
        if isinstance(state, str):
            if state not in STATE_CODES:
                raise ValueError(
                    f"tier_map maps {value!r} to unknown state {state!r}; "
                    f"expected one of {sorted(STATE_CODES)}."
                )
            resolved[int(value)] = STATE_CODES[state]
        else:
            code = int(state)
            if code not in known:
                raise ValueError(
                    f"tier_map maps {value!r} to unknown state code {code}; "
                    f"expected one of {sorted(known)}."
                )
            resolved[int(value)] = code
    return resolved


def _step_dates(da: xr.DataArray, time_dim: str) -> np.ndarray:
    """The step coordinate as datetime64[ns], one date per step."""
    if time_dim not in da.dims:
        raise ValueError(f"'{time_dim}' is not a dimension of da ({da.dims}).")
    if time_dim not in da.coords:
        raise ValueError(
            f"'{time_dim}' has no coordinate values; the step dates come from "
            "the coordinate, so it must carry one."
        )
    values = da[time_dim].values
    # An integer step index would parse as nanoseconds since the epoch, so only
    # datetime and string coordinates are accepted.
    if values.dtype.kind in "OUS":
        try:
            values = pd.to_datetime(values).to_numpy(dtype="datetime64[ns]")
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"The '{time_dim}' coordinate is not date-like, so a step has "
                "no date to record. Give it the start date of each period."
            ) from error
    if values.dtype.kind != "M":
        raise ValueError(
            f"The '{time_dim}' coordinate is not date-like (dtype "
            f"{values.dtype}), so a step has no date to record. Give it the "
            "start date of each period."
        )
    return values.astype("datetime64[ns]")


def _tier_state_block(
    codes: np.ndarray,
    *,
    mapping: dict[int, int],
    fill_value: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map one chunk of tier codes to (state, written). Pure NumPy, chunk-local."""
    codes = np.asarray(codes)
    written = np.zeros(codes.shape, dtype=bool)
    state = np.zeros(codes.shape, dtype="uint8")
    for value, state_code in mapping.items():
        if state_code == STATE_CODES["none"]:
            continue
        match = codes == value
        written |= match
        state = np.where(match, state_code, state).astype("uint8")
    if fill_value is not None:
        is_fill = codes == fill_value
        written &= ~is_fill
        state = np.where(is_fill, 0, state).astype("uint8")
    return state, written


def _collapse_tier_steps_block(
    codes: np.ndarray,
    *,
    mapping: dict[int, int],
    fill_value: int | None,
    step_dates: np.ndarray,
    confirmed_state: int | None,
    base_precision: int,
    class_hint: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse one chunk's step axis to its latest written step.

    ``codes`` arrives with the step axis last, as apply_ufunc places a core
    dimension. The reduction runs over that axis alone, so a chunk never needs
    a neighbour's pixels.
    """
    state_at, written = _tier_state_block(codes, mapping=mapping, fill_value=fill_value)

    n_written = written.sum(axis=-1)
    any_written = n_written > 0
    n_steps = codes.shape[-1]
    latest = n_steps - 1 - np.argmax(written[..., ::-1], axis=-1)
    latest = np.where(any_written, latest, 0)

    state = np.take_along_axis(state_at, latest[..., None], axis=-1)[..., 0]
    state = np.where(any_written, state, 0).astype("uint8")

    first = np.where(any_written, step_dates[latest], _NAT)
    if confirmed_state is None:
        confirm = np.full(state.shape, _NAT, dtype="datetime64[ns]")
    else:
        confirm = np.where(state == confirmed_state, first, _NAT)

    precision = np.where(any_written, base_precision, 0).astype("uint8")
    hint = np.where(any_written, class_hint, 0).astype("uint8")
    return state, first, confirm, precision, hint, n_written > 1


def observations_from_tier_steps(
    da: xr.DataArray,
    *,
    tier_map: Mapping[int, str | int],
    time_dim: str = "time",
    fill_value: int | None = None,
    date_precision: str = "month",
    class_hint: int = 0,
    snapshot_id: str | None = None,
    source: str = "unknown",
    sensor: str = "unknown",
    pixel_size_m: float | None = None,
) -> xr.Dataset:
    """Normalize a stepped layer that writes each pixel at one step only.

    The input is a regularly stepped categorical cube -- monthly, annual or any
    other fixed period -- in which a detected pixel carries a small tier code at
    **one** step and 0 or a fill value at every other. The tier codes say how
    firm the detection is rather than when it happened; the step supplies the
    date. Different layers spell the same tiers with different integers, so
    ``tier_map`` routes a layer's own vocabulary onto :data:`STATE_CODES`.

    The ``time`` dimension is **collapsed**: one write per pixel is one episode,
    so a plain 2-D record holds it without loss. (Where a source's episodes can
    overlap, as with :func:`observations_from_annual_alert_days`, the dimension
    is kept instead.) Use :func:`observations_from_tier_steps_by_step` for the
    per-step masks.

    Parameters
    ----------
    da : xr.DataArray
        Categorical raster with a step dimension and a rioxarray CRS. May be
        dask-backed; the collapse runs chunk by chunk over the step axis and
        stays lazy.
    tier_map : mapping of int to str or int
        Source value to normalized state, given either as a name from
        :data:`STATE_CODES` (``"confirmed"``) or as its code (``2``). Values
        absent from the mapping, and values mapped to ``"none"``, count as
        nothing written, exactly like 0.
    time_dim : str, default "time"
        Name of the step dimension. Its coordinate supplies the dates, so it
        must be date-like and hold the **start** of each period.
    fill_value : int, optional
        Value meaning "no data". It counts as nothing written, and takes
        precedence over ``tier_map`` where the two name the same value.
    date_precision : {"unknown", "day", "month", "lower_bound"}, default "month"
        Precision to record for every date. A monthly step names a month, not a
        day, so the default is ``"month"``.
    class_hint : int, default 0
        Class hint written into ``class_hint`` wherever a detection was found.
        One layer carries one tier vocabulary, so the hint is a constant.
    snapshot_id : str, optional
        Identifier of the snapshot the cube was read from.
    source, sensor : str
        Free-text provenance recorded on the dataset.
    pixel_size_m : float, optional
        Pixel size in metres; inferred from the transform when omitted.

    Returns
    -------
    xr.Dataset
        The normalized record on the input's ``(y, x)`` grid, carrying the
        variables of :data:`OBSERVATION_FIELDS` plus a boolean ``multi_step``.
        ``first_obs_date`` is the step's coordinate value, ``confirm_date``
        repeats it where the state is confirmed, ``last_obs_date`` is NaT and
        ``has_counts`` is False. ``multi_step`` is True at a pixel written at
        more than one step: the latest write is the one kept, and the flag is
        what makes the others visible.

    Raises
    ------
    ValueError
        If ``time_dim`` is not a dimension of ``da`` or carries no date-like
        coordinate, ``tier_map`` is empty or names an unknown state, or
        ``date_precision`` is unknown.
    """
    if date_precision not in DATE_PRECISION:
        raise ValueError(
            f"date_precision must be one of {sorted(DATE_PRECISION)}, "
            f"got {date_precision!r}"
        )
    mapping = _normalize_tier_map(tier_map)
    step_dates = _step_dates(da, time_dim)

    # The reduction reads every step of a pixel, so the step axis must sit in
    # one chunk. It is the short axis; the spatial chunking is left alone.
    if da.chunks is not None:
        da = da.chunk({time_dim: -1})

    state, first, confirm, precision, hint, multi_step = xr.apply_ufunc(
        _collapse_tier_steps_block,
        da,
        input_core_dims=[[time_dim]],
        kwargs={
            "mapping": mapping,
            "fill_value": None if fill_value is None else int(fill_value),
            "step_dates": step_dates,
            "confirmed_state": STATE_CODES.get("confirmed"),
            "base_precision": DATE_PRECISION[date_precision],
            "class_hint": int(class_hint),
        },
        output_core_dims=[[], [], [], [], [], []],
        dask="parallelized",
        output_dtypes=[
            "uint8",
            "datetime64[ns]",
            "datetime64[ns]",
            "uint8",
            "uint8",
            "bool",
        ],
    )

    if pixel_size_m is None:
        pixel_size_m = _infer_pixel_size_m(da)

    variables = {
        "state": state,
        "first_obs_date": first,
        "confirm_date": confirm,
        "last_obs_date": _nat_like(state),
        "n_obs": _constant(state, 0, "uint16"),
        "n_positive": _constant(state, 0, "uint16"),
        "confidence_native": _constant(state, np.nan, "float32"),
        "confidence": _constant(state, np.nan, "float32"),
        "class_hint": hint,
        "date_precision": precision,
        "multi_step": multi_step,
    }
    return _assemble(
        variables,
        crs=da.rio.crs,
        attrs=_record_attrs(
            source=source,
            sensor=sensor,
            pixel_size_m=pixel_size_m,
            snapshot_id=snapshot_id,
            has_counts=False,
        ),
    )


def observations_from_tier_steps_by_step(
    da: xr.DataArray,
    *,
    tier_map: Mapping[int, str | int],
    time_dim: str = "time",
    fill_value: int | None = None,
    date_precision: str = "month",
    class_hint: int = 0,
    snapshot_id: str | None = None,
    source: str = "unknown",
    sensor: str = "unknown",
    pixel_size_m: float | None = None,
) -> xr.Dataset:
    """Normalize a stepped tier layer **without** collapsing the step dimension.

    Same input and same vocabulary handling as
    :func:`observations_from_tier_steps`, but every step keeps its own slice, so
    a caller can mask one period at a time. The mapping is elementwise, so this
    form is chunk-local along the step axis too.

    Parameters
    ----------
    da, tier_map, time_dim, fill_value, date_precision, class_hint, snapshot_id,
    source, sensor, pixel_size_m
        As for :func:`observations_from_tier_steps`.

    Returns
    -------
    xr.Dataset
        The normalized record on the ``(time, y, x)`` source grid. Each step's
        dates are that step's coordinate value. There is no ``multi_step``
        variable: nothing is collapsed here, so no write is hidden.

    Raises
    ------
    ValueError
        As for :func:`observations_from_tier_steps`.
    """
    if date_precision not in DATE_PRECISION:
        raise ValueError(
            f"date_precision must be one of {sorted(DATE_PRECISION)}, "
            f"got {date_precision!r}"
        )
    mapping = _normalize_tier_map(tier_map)
    _step_dates(da, time_dim)

    state, written = xr.apply_ufunc(
        _tier_state_block,
        da,
        kwargs={
            "mapping": mapping,
            "fill_value": None if fill_value is None else int(fill_value),
        },
        output_core_dims=[[], []],
        dask="parallelized",
        output_dtypes=["uint8", "bool"],
    )

    times = da[time_dim].astype("datetime64[ns]")
    first = xr.where(written, times, _NAT).astype("datetime64[ns]")
    confirm = xr.where(state == STATE_CODES["confirmed"], first, _NAT).astype(
        "datetime64[ns]"
    )

    if pixel_size_m is None:
        pixel_size_m = _infer_pixel_size_m(da)

    variables = {
        "state": state,
        "first_obs_date": first,
        "confirm_date": confirm,
        "last_obs_date": _nat_like(state),
        "n_obs": _constant(state, 0, "uint16"),
        "n_positive": _constant(state, 0, "uint16"),
        "confidence_native": _constant(state, np.nan, "float32"),
        "confidence": _constant(state, np.nan, "float32"),
        "class_hint": xr.where(written, int(class_hint), 0).astype("uint8"),
        "date_precision": xr.where(
            written, DATE_PRECISION[date_precision], 0
        ).astype("uint8"),
    }
    return _assemble(
        variables,
        crs=da.rio.crs,
        attrs=_record_attrs(
            source=source,
            sensor=sensor,
            pixel_size_m=pixel_size_m,
            snapshot_id=snapshot_id,
            has_counts=False,
        ),
    )


def select_step_month(ds_by_step: xr.Dataset, year: int, month: int) -> xr.DataArray:
    """Boolean mask of the detections in a per-step record written in a month.

    Parameters
    ----------
    ds_by_step : xr.Dataset
        A normalized record that kept its step dimension, typically from
        :func:`observations_from_tier_steps_by_step`.
    year, month : int
        Calendar year and month number (1-12) to select.

    Returns
    -------
    xr.DataArray
        A boolean array shaped like ``ds_by_step.state``, True where a
        detection was written and its ``first_obs_date`` falls in that month.
    """
    state = ds_by_step["state"]
    dates = ds_by_step["first_obs_date"]
    return (
        (state != STATE_CODES["none"])
        & (state != STATE_FILL)
        & (dates.dt.year == year)
        & (dates.dt.month == month)
    )


# ---------------------------------------------------------------------------
# Table form
# ---------------------------------------------------------------------------


def _finalize_table(table: pd.DataFrame) -> pd.DataFrame:
    """Order the columns and cast each to its schema dtype."""
    ordered = table[list(OBSERVATION_TABLE_DTYPES)]
    return ordered.astype(OBSERVATION_TABLE_DTYPES)


def observations_to_table(ds: xr.Dataset) -> pd.DataFrame:
    """Flatten a normalized record to one row per detected pixel.

    Pixels with ``state`` 0 (no detection) or 255 (fill) are not rows -- a
    record only ever lists detections. A record that kept its ``time``
    dimension contributes one row per detected pixel per step, distinguished by
    the dates rather than by a separate column.

    Parameters
    ----------
    ds : xr.Dataset
        A normalized record from one of the raster adapters.

    Returns
    -------
    pandas.DataFrame
        The flat table described by :data:`OBSERVATION_TABLE_DTYPES`: the
        schema fields, the pixel centres ``x`` and ``y`` in the dataset's CRS,
        and the ``source``, ``sensor``, ``pixel_size_m`` and ``snapshot_id``
        provenance read from the dataset attributes.

    Raises
    ------
    ValueError
        If ``ds`` is missing any of the schema variables.
    """
    missing = [field.name for field in OBSERVATION_FIELDS if field.name not in ds]
    if missing:
        raise ValueError(f"ds is missing observation variable(s) {missing}.")

    y_dim, x_dim = _spatial_dims(ds)
    state = ds["state"].compute() if ds["state"].chunks else ds["state"]
    keep = ((state != 0) & (state != STATE_FILL)).stack(_cell=state.dims)
    index = np.flatnonzero(np.asarray(keep.values))

    columns: dict[str, Any] = {}
    for field in OBSERVATION_FIELDS:
        flat = ds[field.name].transpose(*state.dims).values.reshape(-1)
        columns[field.name] = flat[index]

    y_values = ds[y_dim].values
    x_values = ds[x_dim].values
    mesh_y, mesh_x = np.meshgrid(y_values, x_values, indexing="ij")
    if len(state.dims) > 2:
        repeats = int(np.prod(state.shape) // (len(y_values) * len(x_values)))
        mesh_y = np.tile(mesh_y.reshape(-1), repeats)
        mesh_x = np.tile(mesh_x.reshape(-1), repeats)
    columns["y"] = np.asarray(mesh_y).reshape(-1)[index]
    columns["x"] = np.asarray(mesh_x).reshape(-1)[index]

    n = len(index)
    columns["source"] = np.full(n, ds.attrs.get("source", "unknown"), dtype=object)
    columns["sensor"] = np.full(n, ds.attrs.get("sensor", "unknown"), dtype=object)
    columns["pixel_size_m"] = np.full(
        n, float(ds.attrs.get("pixel_size_m", np.nan)), dtype="float64"
    )
    columns["snapshot_id"] = np.full(
        n, str(ds.attrs.get("snapshot_id", "")), dtype=object
    )

    return _finalize_table(pd.DataFrame(columns))


def write_observations(table: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Write an observation table to Parquet.

    pyarrow does the writing -- :func:`read_observations` reads with it too,
    and a categorical column only survives a round trip when both ends use the
    same engine. Categorical columns are stored as a Parquet dictionary and
    come back as pandas categoricals; ``datetime64[ns]`` columns keep NaT.

    Parameters
    ----------
    table : pandas.DataFrame
        A table from :func:`observations_to_table` or
        :func:`observations_from_points`.
    path : str or path-like
        Destination file.

    Raises
    ------
    ValueError
        If ``table`` does not carry the schema columns.
    """
    _require_pyarrow()
    missing = [col for col in OBSERVATION_TABLE_DTYPES if col not in table.columns]
    if missing:
        raise ValueError(f"table is missing observation column(s) {missing}.")
    table.to_parquet(path, index=False)


def read_observations(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read an observation table written by :func:`write_observations`.

    Parameters
    ----------
    path : str or path-like
        Source file.

    Returns
    -------
    pandas.DataFrame
        The table with its schema column order and dtypes restored. Table
        attributes such as a point table's CRS are not stored in Parquet; pass
        the CRS explicitly to :func:`points_to_grid_mask` afterwards.
    """
    _require_pyarrow()
    return _finalize_table(pd.read_parquet(path))


# ---------------------------------------------------------------------------
# Bridge to vector patches
# ---------------------------------------------------------------------------


def observations_mask(
    ds: xr.Dataset,
    states: Sequence[int] = (2,),
    month: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Reduce a normalized record to a 2-D boolean mask of selected detections.

    The result is what
    :func:`~ctreeskit.xr_analyzer.xr_vectorize_module.patches_from_mask`
    expects: a 2-D boolean raster carrying the record's CRS and transform. A
    record that kept a ``time`` dimension is reduced with "any step", so a pixel
    detected in any retained step is True.

    Parameters
    ----------
    ds : xr.Dataset
        A normalized record.
    states : sequence of int, default (2,)
        State codes to keep. The default keeps confirmed detections only.
    month : tuple of (int, int), optional
        Year and month number. When given, only detections whose
        ``first_obs_date`` falls in that month are kept.

    Returns
    -------
    xr.DataArray
        A 2-D boolean ``(y, x)`` mask with the record's CRS.
    """
    selected = ds["state"].isin(np.asarray(list(states), dtype="uint8"))
    if month is not None:
        year, month_number = month
        dates = ds["first_obs_date"]
        selected = selected & (dates.dt.year == year) & (dates.dt.month == month_number)

    y_dim, x_dim = _spatial_dims(ds)
    extra = [dim for dim in selected.dims if dim not in (y_dim, x_dim)]
    if extra:
        selected = selected.any(dim=extra)

    mask = selected.transpose(y_dim, x_dim).astype(bool)
    crs = ds.rio.crs
    if crs is not None:
        mask = mask.rio.write_crs(crs)
        mask.rio.write_transform(ds.rio.transform(), inplace=True)
    return mask


__all__ = [
    "STATE_CODES",
    "STATE_FILL",
    "DATE_PRECISION",
    "DEFAULT_CODE_OFFSETS",
    "ObservationField",
    "OBSERVATION_FIELDS",
    "OBSERVATION_ATTRS",
    "OBSERVATION_TABLE_DTYPES",
    "observations_from_dated_codes",
    "observations_from_annual_alert_days",
    "select_month",
    "observations_from_tier_steps",
    "observations_from_tier_steps_by_step",
    "select_step_month",
    "observations_from_points",
    "points_to_grid_mask",
    "observations_to_table",
    "write_observations",
    "read_observations",
    "observations_mask",
]

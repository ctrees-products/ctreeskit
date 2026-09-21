"""Tests for xr_observations_module, over small synthetic layers.

Grids use realistic WGS 84 coordinates (descending latitude, north-up) or a
metre-based projected CRS, so the coordinate and pixel-size paths are
exercised for real.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("pyarrow")

from ctreeskit import (  # noqa: E402
    DATE_PRECISION,
    OBSERVATION_FIELDS,
    OBSERVATION_TABLE_DTYPES,
    STATE_CODES,
    observations_from_annual_alert_days,
    observations_from_dated_codes,
    observations_from_points,
    observations_from_tier_steps,
    observations_from_tier_steps_by_step,
    observations_mask,
    observations_to_table,
    patches_from_mask,
    points_to_grid_mask,
    read_observations,
    select_month,
    select_step_month,
    write_observations,
)

# ~5 m at the equator (one degree of latitude is about 111 km).
DEG_5M = 4.5e-5


def _projected_raster(values, res=30.0, x0=500000.0, y0=4000000.0, crs="EPSG:32633"):
    """Wrap an array whose last two axes are (y, x) on a north-up metre grid."""
    values = np.asarray(values)
    ny, nx = values.shape[-2:]
    x = x0 + (np.arange(nx) + 0.5) * res
    y = y0 - (np.arange(ny) + 0.5) * res
    if values.ndim == 2:
        da = xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])
    else:
        da = xr.DataArray(
            values,
            coords={"time": np.arange(values.shape[0]) + 2024, "y": y, "x": x},
            dims=["time", "y", "x"],
        )
    return da.rio.write_crs(crs)


def _geographic_raster(values, res=DEG_5M, lon0=-60.0, lat0=0.0):
    """Wrap a 2-D array on a north-up WGS 84 grid at the given resolution."""
    values = np.asarray(values)
    ny, nx = values.shape
    x = lon0 + (np.arange(nx) + 0.5) * res
    y = lat0 - (np.arange(ny) + 0.5) * res
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])
    return da.rio.write_crs("EPSG:4326")


FILL = 255
STABLE = 1


def _coded_grid():
    """A 6x6 grid of dated state codes: confirmed, candidate, rejected, stable, fill.

    Confirmed codes are bare ``YYMM``; candidates carry +10000 and rejections
    +20000. Detections sit in 2024-03, 2024-05 and 2024-11.
    """
    grid = np.full((6, 6), STABLE, dtype="int32")
    grid[0, 0:2] = 2403           # confirmed, March 2024
    grid[1, 0] = 2411             # confirmed, November 2024
    grid[2, 2:4] = 10000 + 2405   # candidate, May 2024
    grid[3, 4] = 20000 + 2403     # rejected, March 2024
    grid[4:6, 4:6] = FILL
    return grid


class TestDatedCodes:
    def test_states_and_dates_decode(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()),
            stable_values=(STABLE,),
            fill_value=FILL,
            source="raster_codes",
            sensor="optical",
            snapshot_id="snap-1",
        )
        state = ds["state"].values
        assert state[0, 0] == STATE_CODES["confirmed"]
        assert state[2, 2] == STATE_CODES["candidate"]
        assert state[3, 4] == STATE_CODES["rejected"]
        assert state[5, 0] == STATE_CODES["none"]   # stable
        assert state[5, 5] == STATE_CODES["none"]   # fill
        assert (state != 0).sum() == 6

        first = ds["first_obs_date"].values
        assert first[0, 0] == np.datetime64("2024-03-01")
        assert first[1, 0] == np.datetime64("2024-11-01")
        assert first[2, 2] == np.datetime64("2024-05-01")
        assert np.isnat(first[5, 0])
        assert np.isnat(first[5, 5])

    def test_confirm_date_only_on_confirmed_pixels(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()), stable_values=(STABLE,), fill_value=FILL
        )
        confirm = ds["confirm_date"].values
        assert confirm[0, 0] == np.datetime64("2024-03-01")
        assert np.isnat(confirm[2, 2])   # candidate
        assert np.isnat(confirm[3, 4])   # rejected
        assert np.isnat(ds["last_obs_date"].values).all()

    def test_precision_is_month_by_default(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()), stable_values=(STABLE,), fill_value=FILL
        )
        precision = ds["date_precision"].values
        assert precision[0, 0] == DATE_PRECISION["month"]
        assert precision[2, 2] == DATE_PRECISION["month"]
        assert precision[5, 0] == DATE_PRECISION["unknown"]

    def test_lower_bound_months_flags_only_the_chosen_month(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()),
            stable_values=(STABLE,),
            fill_value=FILL,
            lower_bound_months=[3],
        )
        precision = ds["date_precision"].values
        assert precision[0, 0] == DATE_PRECISION["lower_bound"]   # March confirmed
        assert precision[3, 4] == DATE_PRECISION["lower_bound"]   # March rejected
        assert precision[1, 0] == DATE_PRECISION["month"]         # November
        assert precision[2, 2] == DATE_PRECISION["month"]         # May

    def test_four_digit_year_format(self):
        grid = np.full((2, 2), STABLE, dtype="int32")
        grid[0, 0] = 202403
        ds = observations_from_dated_codes(
            _projected_raster(grid),
            epoch_format="YYYYMM",
            offsets={"confirmed": 0, "candidate": 1_000_000},
            stable_values=(STABLE,),
        )
        assert ds["first_obs_date"].values[0, 0] == np.datetime64("2024-03-01")

    def test_class_hint_map_marks_the_band(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()),
            stable_values=(STABLE,),
            fill_value=FILL,
            class_hint_map={"rejected": 7},
        )
        assert ds["class_hint"].values[3, 4] == 7
        assert ds["class_hint"].values[0, 0] == 0

    def test_pixel_size_inferred_from_a_metre_grid(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid(), res=30.0), stable_values=(STABLE,)
        )
        assert ds.attrs["pixel_size_m"] == pytest.approx(30.0)
        assert ds.attrs["has_counts"] is False
        assert ds.rio.crs.to_epsg() == 32633

    def test_pixel_size_is_nan_on_a_geographic_grid(self):
        ds = observations_from_dated_codes(
            _geographic_raster(_coded_grid()), stable_values=(STABLE,)
        )
        assert np.isnan(ds.attrs["pixel_size_m"])

    def test_rejects_a_time_dimension(self):
        cube = _projected_raster(_coded_grid()).expand_dims(time=[2024])
        with pytest.raises(ValueError, match="2-D"):
            observations_from_dated_codes(cube)

    def test_rejects_an_offset_off_the_band_width(self):
        with pytest.raises(ValueError, match="band width"):
            observations_from_dated_codes(
                _projected_raster(_coded_grid()), offsets={"confirmed": 500}
            )

    def test_rejects_an_unknown_epoch_format(self):
        with pytest.raises(ValueError, match="epoch_format"):
            observations_from_dated_codes(
                _projected_raster(_coded_grid()), epoch_format="MMYY"
            )


class _CountingSource:
    """A numpy-array facade that records how many blocks dask pulled from it."""

    def __init__(self, values):
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.ndim = values.ndim
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self._values[key]


class TestDatedCodesLaziness:
    def _chunked(self):
        import dask.array as dsa

        source = _CountingSource(_coded_grid())
        da = _projected_raster(_coded_grid()).copy(
            data=dsa.from_array(source, chunks=(3, 3))
        )
        # from_array probes the source once while building the graph; the test
        # counts reads caused by the decode, so start from zero.
        source.reads = 0
        return da, source

    def test_decode_stays_lazy(self):
        da, source = self._chunked()
        ds = observations_from_dated_codes(
            da, stable_values=(STABLE,), fill_value=FILL
        )
        assert ds["state"].chunks is not None
        assert ds["first_obs_date"].chunks is not None
        assert source.reads == 0

    def test_one_chunk_read_for_one_chunk_of_output(self):
        da, source = self._chunked()
        ds = observations_from_dated_codes(
            da, stable_values=(STABLE,), fill_value=FILL
        )
        corner = ds["state"].isel(y=slice(0, 3), x=slice(0, 3)).compute()

        assert source.reads == 1     # the other three chunks were never touched
        assert corner.values[0, 0] == STATE_CODES["confirmed"]


def _alert_cube():
    """Two annual steps with one alert each, on a 4x4 30 m projected grid."""
    alert = np.zeros((2, 4, 4), dtype="uint8")
    alert[0, 1, 1] = 1
    alert[1, 2, 2] = 3          # a non-zero value that also encodes a class
    day = np.full((2, 4, 4), -9999, dtype="int32")
    day[0, 1, 1] = (np.datetime64("2024-03-15") - np.datetime64("1970-01-01")).astype(
        "int64"
    )
    day[1, 2, 2] = (np.datetime64("2025-07-02") - np.datetime64("1970-01-01")).astype(
        "int64"
    )
    return _projected_raster(alert), _projected_raster(day)


class TestAnnualAlertDays:
    def test_days_decode_inside_their_year(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day, source="alerts")

        first = ds["first_obs_date"].values
        assert first[0, 1, 1] == np.datetime64("2024-03-15")
        assert first[1, 2, 2] == np.datetime64("2025-07-02")
        assert pd.Timestamp(first[0, 1, 1]).year == int(ds["time"].values[0])
        assert pd.Timestamp(first[1, 2, 2]).year == int(ds["time"].values[1])

    def test_time_dimension_is_kept(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)
        assert ds["state"].dims == ("time", "y", "x")
        assert ds.sizes["time"] == 2

    def test_all_three_dates_are_the_alert_day(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)
        for name in ("first_obs_date", "confirm_date", "last_obs_date"):
            assert ds[name].values[0, 1, 1] == np.datetime64("2024-03-15")

    def test_state_and_precision(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)
        assert ds["state"].values[0, 1, 1] == STATE_CODES["alert"]
        assert ds["state"].values[0, 0, 0] == STATE_CODES["none"]
        assert ds["date_precision"].values[1, 2, 2] == DATE_PRECISION["day"]
        assert ds["date_precision"].values[0, 0, 0] == DATE_PRECISION["unknown"]

    def test_class_hint_from_the_alert_value(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day, class_hint_map={3: 4})
        assert ds["class_hint"].values[1, 2, 2] == 4
        assert ds["class_hint"].values[0, 1, 1] == 0

    def test_alert_band_fill_is_not_an_alert(self):
        """A band whose nodata is a non-zero sentinel must not decode to alerts.

        The sentinel is non-zero, so "non-zero marks an alert" would read every
        filled pixel as a detection wherever the date band happens to be
        readable there. ``alert_fill_value`` is what keeps those pixels at
        state 0.
        """
        fill = -9999
        values = np.full((2, 4, 4), fill, dtype="int16")
        values[0, 1, 1] = 1
        alert = _projected_raster(values)
        # Every pixel carries a readable date, so only the alert band can tell
        # the one detection apart from the fill.
        day = _projected_raster(
            np.full(
                (2, 4, 4),
                (np.datetime64("2024-03-15") - np.datetime64("1970-01-01")).astype(
                    "int64"
                ),
                dtype="int32",
            )
        )

        without = observations_from_annual_alert_days(alert, day)
        assert (without["state"].values == STATE_CODES["alert"]).sum() == 32

        ds = observations_from_annual_alert_days(alert, day, alert_fill_value=fill)
        state = ds["state"].values
        assert (state == STATE_CODES["alert"]).sum() == 1
        assert state[0, 1, 1] == STATE_CODES["alert"]
        assert (state[values == fill] == STATE_CODES["none"]).all()
        assert (
            ds["date_precision"].values[values == fill] == DATE_PRECISION["unknown"]
        ).all()
        assert pd.isna(ds["first_obs_date"].values[values == fill]).all()

    def test_non_epoch_day_offsets(self):
        alert, _ = _alert_cube()
        day = _projected_raster(np.full((2, 4, 4), -9999, dtype="int32"))
        day.values[0, 1, 1] = 74      # 74 days after 2024-01-01
        ds = observations_from_annual_alert_days(alert, day, day_epoch="2024-01-01")
        assert ds["first_obs_date"].values[0, 1, 1] == np.datetime64("2024-03-15")

    def test_select_month_picks_the_right_pixels(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)

        march = select_month(ds, 2024, 3)
        assert march.values.sum() == 1
        assert march.values[0, 1, 1]

        july = select_month(ds, 2025, 7)
        assert july.values.sum() == 1
        assert july.values[1, 2, 2]

        assert select_month(ds, 2024, 4).values.sum() == 0

    def test_rejects_mismatched_inputs(self):
        alert, day = _alert_cube()
        with pytest.raises(ValueError, match="share a shape"):
            observations_from_annual_alert_days(alert, day.isel(x=slice(0, 3)))
        with pytest.raises(ValueError, match="not a dimension of alert"):
            observations_from_annual_alert_days(alert.isel(time=0), day)


def _point_frame():
    """Four point detections near the equator, with both confidence flavours."""
    return pd.DataFrame(
        {
            # Cell centres of a 5 m grid starting at (-60.0, 0.0), plus one
            # point far off that grid.
            "lon": [-60.0 + c * DEG_5M for c in (0.5, 2.5, 4.5)] + [-59.5],
            "lat": [-r * DEG_5M for r in (0.5, 2.5, 4.5)] + [-0.5],
            "acquired": [
                "2024-03-01T12:30:00",
                "2024-03-01T13:00:00",
                "2024-03-02T01:15:00",
                "2024-03-03T02:00:00",
            ],
            "confidence": ["low", "nominal", "high", "high"],
            "confidence_pct": [35, 60, 88, 91],
            "platform": ["sensor-a", "sensor-a", "sensor-b", "sensor-b"],
        }
    )


CONFIDENCE_MAP = {"low": 0.2, "nominal": 0.6, "high": 0.9}


class TestPoints:
    def test_categorical_confidence_is_mapped(self):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            confidence_col="confidence",
            confidence_map=CONFIDENCE_MAP,
            crs="EPSG:4326",
            source="points",
            sensor="thermal",
        )
        assert table["confidence"].to_list() == pytest.approx([0.2, 0.6, 0.9, 0.9])
        assert table["confidence_native"].isna().all()
        assert (table["state"] == STATE_CODES["point_detection"]).all()
        assert (table["date_precision"] == DATE_PRECISION["day"]).all()
        assert table["first_obs_date"].iloc[0] == pd.Timestamp("2024-03-01T12:30:00")

    def test_numeric_percentage_is_normalized(self):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            confidence_col="confidence_pct",
            crs="EPSG:4326",
            source="points",
        )
        assert table["confidence_native"].to_list() == pytest.approx([35, 60, 88, 91])
        assert table["confidence"].to_list() == pytest.approx(
            [0.35, 0.60, 0.88, 0.91], abs=1e-6
        )
        assert table["confidence"].between(0, 1).all()

    def test_sensor_column_overrides_the_constant(self):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            sensor_col="platform",
            crs="EPSG:4326",
        )
        assert table["sensor"].to_list() == [
            "sensor-a",
            "sensor-a",
            "sensor-b",
            "sensor-b",
        ]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="no column"):
            observations_from_points(
                _point_frame(),
                x_col="lon",
                y_col="lat",
                datetime_col="not_there",
                crs="EPSG:4326",
            )

    def test_rasterize_onto_a_reference_grid(self):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            crs="EPSG:4326",
            source="points",
        )
        reference = _geographic_raster(np.zeros((10, 10), dtype="uint8"))
        mask = points_to_grid_mask(table, reference)

        assert mask.dtype == bool
        assert mask.shape == (10, 10)
        # Three points land on the grid at 5 m spacing; the fourth is far off it.
        assert sorted(map(tuple, np.argwhere(mask.values))) == [(0, 0), (2, 2), (4, 4)]
        assert mask.rio.crs.to_epsg() == 4326

    def test_rasterize_reprojects_when_the_crs_differs(self):
        frame = pd.DataFrame(
            {
                "easting": [500045.0, 500105.0],
                "northing": [3999985.0, 3999925.0],
                "acquired": ["2024-03-01", "2024-03-02"],
            }
        )
        table = observations_from_points(
            frame,
            x_col="easting",
            y_col="northing",
            datetime_col="acquired",
            crs="EPSG:32633",
        )
        reference = _projected_raster(np.zeros((4, 4), dtype="uint8"), res=30.0)
        direct = points_to_grid_mask(table, reference)
        assert sorted(map(tuple, np.argwhere(direct.values))) == [(0, 1), (2, 3)]

        geographic = table.copy()
        transformed = reference.rio.reproject("EPSG:4326")
        reprojected = points_to_grid_mask(geographic, transformed)
        assert reprojected.values.sum() == 2

    def test_rasterize_needs_a_crs_after_a_round_trip(self, tmp_path):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            crs="EPSG:4326",
        )
        path = tmp_path / "points.parquet"
        write_observations(table, path)
        restored = read_observations(path)

        reference = _geographic_raster(np.zeros((10, 10), dtype="uint8"))
        mask = points_to_grid_mask(restored, reference, crs="EPSG:4326")
        assert mask.values.sum() == 3

    def test_rasterize_without_any_crs_raises(self):
        table = observations_from_points(
            _point_frame(),
            x_col="lon",
            y_col="lat",
            datetime_col="acquired",
            crs="EPSG:4326",
        )
        table.attrs.pop("crs")
        reference = _geographic_raster(np.zeros((10, 10), dtype="uint8"))
        with pytest.raises(ValueError, match="carries no CRS"):
            points_to_grid_mask(table, reference)


def _monthly_raster(values, res=30.0, x0=500000.0, y0=4000000.0, crs="EPSG:32633"):
    """Wrap a (step, y, x) array on a north-up metre grid, stepped by month."""
    values = np.asarray(values)
    nt, ny, nx = values.shape
    x = x0 + (np.arange(nx) + 0.5) * res
    y = y0 - (np.arange(ny) + 0.5) * res
    time = pd.date_range("2024-01-01", periods=nt, freq="MS")
    da = xr.DataArray(
        values, coords={"time": time, "y": y, "x": x}, dims=["time", "y", "x"]
    )
    return da.rio.write_crs(crs)


TIER_FILL = 255

#: Two layers spell the same tiers with different integers.
LOSS_TIERS = {1: "confirmed", 2: "candidate", 3: "rejected"}
DETECTION_TIERS = {3: "confirmed", 1: "candidate", 2: "rejected"}


def _tier_cube():
    """Three monthly steps, each detected pixel written at exactly one step.

    Pixel (0, 0) is written in January, (1, 1) in February and (2, 2) in
    March, one per tier. Pixel (3, 3) is written twice -- February and March --
    to exercise the collapse rule. Everything else is 0 or fill.
    """
    grid = np.zeros((3, 4, 4), dtype="int16")
    grid[0, 0, 0] = 1           # January
    grid[1, 1, 1] = 2           # February
    grid[2, 2, 2] = 3           # March
    grid[1, 3, 3] = 2           # written twice: February ...
    grid[2, 3, 3] = 1           # ... and again in March
    grid[:, 0, 3] = TIER_FILL
    return _monthly_raster(grid)


class TestTierSteps:
    def test_loss_vocabulary_maps_onto_the_states(self):
        ds = observations_from_tier_steps(
            _tier_cube(),
            tier_map=LOSS_TIERS,
            fill_value=TIER_FILL,
            source="tier_layer",
            sensor="optical",
            snapshot_id="snap-4",
        )
        state = ds["state"].values
        assert state.shape == (4, 4)
        assert state[0, 0] == STATE_CODES["confirmed"]
        assert state[1, 1] == STATE_CODES["candidate"]
        assert state[2, 2] == STATE_CODES["rejected"]
        assert state[0, 1] == STATE_CODES["none"]
        assert state[0, 3] == STATE_CODES["none"]   # fill
        assert ds.attrs["source"] == "tier_layer"
        assert ds.attrs["has_counts"] is False
        assert ds.rio.crs.to_epsg() == 32633

    def test_detection_vocabulary_reaches_the_same_states(self):
        """A layer spelling the tiers with other integers normalizes the same."""
        swapped = _tier_cube()
        loss = observations_from_tier_steps(
            swapped, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        detection = observations_from_tier_steps(
            swapped, tier_map=DETECTION_TIERS, fill_value=TIER_FILL
        )
        assert loss["state"].values[0, 0] == STATE_CODES["confirmed"]
        assert detection["state"].values[0, 0] == STATE_CODES["candidate"]
        assert detection["state"].values[2, 2] == STATE_CODES["confirmed"]
        # Both vocabularies see the same pixels written, at the same steps.
        assert (
            (loss["state"].values != 0) == (detection["state"].values != 0)
        ).all()
        assert (
            loss["first_obs_date"].values == detection["first_obs_date"].values
        ).all() or np.isnat(loss["first_obs_date"].values).any()

    def test_state_codes_may_be_given_as_integers(self):
        ds = observations_from_tier_steps(
            _tier_cube(),
            tier_map={1: STATE_CODES["confirmed"], 2: 1, 3: 3},
            fill_value=TIER_FILL,
        )
        assert ds["state"].values[0, 0] == STATE_CODES["confirmed"]
        assert ds["state"].values[1, 1] == STATE_CODES["candidate"]

    def test_dates_are_the_start_of_the_step(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        first = ds["first_obs_date"].values
        assert first[0, 0] == np.datetime64("2024-01-01")
        assert first[1, 1] == np.datetime64("2024-02-01")
        assert first[2, 2] == np.datetime64("2024-03-01")
        assert np.isnat(first[0, 1])
        assert np.isnat(first[0, 3])

    def test_confirm_date_only_on_confirmed_pixels(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        confirm = ds["confirm_date"].values
        assert confirm[0, 0] == np.datetime64("2024-01-01")
        assert np.isnat(confirm[1, 1])   # candidate
        assert np.isnat(confirm[2, 2])   # rejected
        assert np.isnat(ds["last_obs_date"].values).all()

    def test_precision_is_month_by_default(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        precision = ds["date_precision"].values
        assert precision[0, 0] == DATE_PRECISION["month"]
        assert precision[0, 1] == DATE_PRECISION["unknown"]

        as_bound = observations_from_tier_steps(
            _tier_cube(),
            tier_map=LOSS_TIERS,
            fill_value=TIER_FILL,
            date_precision="lower_bound",
        )
        assert as_bound["date_precision"].values[0, 0] == DATE_PRECISION["lower_bound"]

    def test_latest_step_wins_and_multi_step_is_flagged(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        multi = ds["multi_step"].values
        assert multi.dtype == bool
        assert multi[3, 3]
        assert multi.sum() == 1
        assert not multi[0, 0]
        # March's code (1 -> confirmed) wins over February's (2 -> candidate).
        assert ds["state"].values[3, 3] == STATE_CODES["confirmed"]
        assert ds["first_obs_date"].values[3, 3] == np.datetime64("2024-03-01")

    def test_class_hint_is_constant_for_the_layer(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL, class_hint=6
        )
        hint = ds["class_hint"].values
        assert hint[0, 0] == 6
        assert hint[1, 1] == 6
        assert hint[0, 1] == 0
        assert hint[0, 3] == 0

    def test_counts_are_absent(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert (ds["n_obs"].values == 0).all()
        assert (ds["n_positive"].values == 0).all()
        assert np.isnan(ds["confidence"].values).all()

    def test_pixel_size_inferred_from_a_metre_grid(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert ds.attrs["pixel_size_m"] == pytest.approx(30.0)

    def test_unmapped_values_count_as_nothing_written(self):
        grid = np.zeros((3, 2, 2), dtype="int16")
        grid[0, 0, 0] = 9        # not in tier_map
        ds = observations_from_tier_steps(_monthly_raster(grid), tier_map=LOSS_TIERS)
        assert (ds["state"].values == 0).all()

    def test_rejects_a_missing_or_undated_step_dimension(self):
        cube = _tier_cube()
        with pytest.raises(ValueError, match="not a dimension"):
            observations_from_tier_steps(
                cube, tier_map=LOSS_TIERS, time_dim="step"
            )
        undated = cube.assign_coords(time=np.arange(cube.sizes["time"]))
        with pytest.raises(ValueError, match="not date-like"):
            observations_from_tier_steps(undated, tier_map=LOSS_TIERS)

    def test_rejects_an_unknown_state_and_an_empty_map(self):
        cube = _tier_cube()
        with pytest.raises(ValueError, match="unknown state"):
            observations_from_tier_steps(cube, tier_map={1: "gone"})
        with pytest.raises(ValueError, match="unknown state code"):
            observations_from_tier_steps(cube, tier_map={1: 99})
        with pytest.raises(ValueError, match="empty"):
            observations_from_tier_steps(cube, tier_map={})

    def test_rejects_an_unknown_precision(self):
        with pytest.raises(ValueError, match="date_precision"):
            observations_from_tier_steps(
                _tier_cube(), tier_map=LOSS_TIERS, date_precision="fortnight"
            )


class TestTierStepsByStep:
    def test_step_dimension_is_kept(self):
        ds = observations_from_tier_steps_by_step(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert ds["state"].dims == ("time", "y", "x")
        assert ds.sizes["time"] == 3
        assert "multi_step" not in ds

    def test_every_write_keeps_its_own_step_and_date(self):
        ds = observations_from_tier_steps_by_step(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        state = ds["state"].values
        assert state[0, 0, 0] == STATE_CODES["confirmed"]
        assert state[1, 0, 0] == STATE_CODES["none"]
        # The double-written pixel keeps both writes here, unlike the collapse.
        assert state[1, 3, 3] == STATE_CODES["candidate"]
        assert state[2, 3, 3] == STATE_CODES["confirmed"]

        first = ds["first_obs_date"].values
        assert first[1, 3, 3] == np.datetime64("2024-02-01")
        assert first[2, 3, 3] == np.datetime64("2024-03-01")
        assert np.isnat(first[0, 3, 3])

    def test_string_step_coordinates_parse_like_the_collapse(self):
        cube = _tier_cube().assign_coords(time=["2024-01-01", "2024-02-01", "2024-03-01"])
        by_step = observations_from_tier_steps_by_step(
            cube, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        collapsed = observations_from_tier_steps(
            cube, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert by_step["first_obs_date"].dtype == np.dtype("datetime64[ns]")
        assert by_step["first_obs_date"].values[2, 2, 2] == np.datetime64("2024-03-01")
        assert (
            by_step["first_obs_date"].values[2, 2, 2]
            == collapsed["first_obs_date"].values[2, 2]
        )

    def test_class_hint_outside_uint8_is_rejected(self):
        with pytest.raises(ValueError, match="0..255"):
            observations_from_tier_steps_by_step(
                _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL, class_hint=300
            )
        with pytest.raises(ValueError, match="0..255"):
            observations_from_tier_steps(
                _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL, class_hint=-1
            )

    def test_select_step_month_picks_one_period(self):
        ds = observations_from_tier_steps_by_step(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        january = select_step_month(ds, 2024, 1)
        assert january.values.sum() == 1
        assert january.values[0, 0, 0]

        march = select_step_month(ds, 2024, 3)
        assert march.values.sum() == 2      # (2, 2) and the second write at (3, 3)

        assert select_step_month(ds, 2024, 4).values.sum() == 0


class TestTierStepsLaziness:
    def _chunked(self):
        import dask.array as dsa

        values = _tier_cube().values
        source = _CountingSource(values)
        da = _tier_cube().copy(data=dsa.from_array(source, chunks=(3, 2, 2)))
        # from_array probes the source once while building the graph.
        source.reads = 0
        return da, source

    def test_collapse_stays_lazy(self):
        da, source = self._chunked()
        ds = observations_from_tier_steps(
            da, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert ds["state"].chunks is not None
        assert ds["first_obs_date"].chunks is not None
        assert ds["multi_step"].chunks is not None
        assert source.reads == 0

    def test_one_chunk_read_for_one_chunk_of_output(self):
        da, source = self._chunked()
        ds = observations_from_tier_steps(
            da, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        corner = ds["state"].isel(y=slice(0, 2), x=slice(0, 2)).compute()

        assert source.reads == 1     # the other three chunks were never touched
        assert corner.values[0, 0] == STATE_CODES["confirmed"]

    def test_by_step_stays_lazy(self):
        da, source = self._chunked()
        ds = observations_from_tier_steps_by_step(
            da, tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert ds["state"].chunks is not None
        assert source.reads == 0


class TestTierStepsBridges:
    def test_collapsed_mask_feeds_patches_from_mask(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        mask = observations_mask(ds)
        assert mask.dims == ("y", "x")
        assert mask.values.sum() == 2       # (0, 0) and the collapsed (3, 3)

        patches = patches_from_mask(mask, connectivity=4)
        assert len(patches) == 2
        assert patches["pixel_count"].to_list() == [1, 1]
        assert patches["area_ha"].iloc[0] == pytest.approx(0.09)
        assert patches.crs.to_epsg() == 32633

    def test_mask_month_filter_uses_the_step_date(self):
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert observations_mask(ds, month=(2024, 1)).values.sum() == 1
        assert observations_mask(ds, month=(2024, 3)).values.sum() == 1
        assert observations_mask(ds, states=(1, 2, 3)).values.sum() == 4

    def test_table_round_trip_keeps_the_schema(self, tmp_path):
        ds = observations_from_tier_steps(
            _tier_cube(),
            tier_map=LOSS_TIERS,
            fill_value=TIER_FILL,
            source="tier_layer",
            sensor="optical",
            snapshot_id="snap-4",
        )
        table = observations_to_table(ds)

        assert list(table.columns) == list(OBSERVATION_TABLE_DTYPES)
        assert len(table) == 4              # the collapsed pixel is one row
        assert set(table["source"].astype(str)) == {"tier_layer"}

        path = tmp_path / "tiers.parquet"
        write_observations(table, path)
        restored = read_observations(path)
        assert [str(d) for d in restored.dtypes] == [str(d) for d in table.dtypes]
        assert restored["first_obs_date"].equals(table["first_obs_date"])
        assert restored["state"].equals(table["state"])

    def test_the_extra_variable_does_not_reach_the_table(self):
        """``multi_step`` rides on the Dataset only; the column set is fixed."""
        ds = observations_from_tier_steps(
            _tier_cube(), tier_map=LOSS_TIERS, fill_value=TIER_FILL
        )
        assert "multi_step" in ds
        assert "multi_step" not in observations_to_table(ds).columns

    def test_tier_rows_concatenate_with_the_other_adapters(self):
        rows = observations_to_table(
            observations_from_tier_steps(
                _tier_cube(),
                tier_map=LOSS_TIERS,
                fill_value=TIER_FILL,
                source="tier_layer",
            )
        )
        combined = pd.concat([*_all_three_tables(), rows], ignore_index=True)
        assert set(combined.columns) == set(OBSERVATION_TABLE_DTYPES)
        assert len(combined) == 6 + 2 + 4 + 4


def _all_three_tables():
    """One table from each adapter, over the same synthetic material."""
    coded = observations_to_table(
        observations_from_dated_codes(
            _projected_raster(_coded_grid()),
            stable_values=(STABLE,),
            fill_value=FILL,
            source="raster_codes",
            sensor="optical",
            snapshot_id="snap-1",
        )
    )
    alert, day = _alert_cube()
    alerts = observations_to_table(
        observations_from_annual_alert_days(
            alert, day, source="alerts", sensor="radar", snapshot_id="snap-2"
        )
    )
    points = observations_from_points(
        _point_frame(),
        x_col="lon",
        y_col="lat",
        datetime_col="acquired",
        confidence_col="confidence",
        confidence_map=CONFIDENCE_MAP,
        crs="EPSG:4326",
        source="points",
        sensor="thermal",
        snapshot_id="snap-3",
    )
    return coded, alerts, points


class TestTableContract:
    def test_every_adapter_yields_the_same_columns_and_dtypes(self):
        tables = _all_three_tables()
        expected_columns = set(OBSERVATION_TABLE_DTYPES)
        expected_dtypes = {
            name: str(dtype) for name, dtype in OBSERVATION_TABLE_DTYPES.items()
        }
        for table in tables:
            assert set(table.columns) == expected_columns
            assert {
                name: str(dtype) for name, dtype in table.dtypes.items()
            } == expected_dtypes

    def test_schema_fields_are_all_columns(self):
        for field in OBSERVATION_FIELDS:
            assert field.name in OBSERVATION_TABLE_DTYPES
            assert OBSERVATION_TABLE_DTYPES[field.name] == field.dtype

    def test_tables_from_different_sources_concatenate(self):
        combined = pd.concat(_all_three_tables(), ignore_index=True)
        assert set(combined["source"].astype(str)) == {
            "raster_codes",
            "alerts",
            "points",
        }
        assert len(combined) == 6 + 2 + 4

    def test_only_detections_become_rows(self):
        coded, alerts, points = _all_three_tables()
        assert (coded["state"] != 0).all()
        assert len(coded) == 6        # stable and fill pixels are not rows
        assert len(alerts) == 2


class TestObservationsMask:
    def test_confirmed_only_by_default(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()), stable_values=(STABLE,), fill_value=FILL
        )
        mask = observations_mask(ds)
        assert mask.dtype == bool
        assert mask.shape == (6, 6)
        assert mask.values.sum() == 3       # three confirmed pixels
        assert mask.rio.crs.to_epsg() == 32633

    def test_month_filter(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()), stable_values=(STABLE,), fill_value=FILL
        )
        assert observations_mask(ds, month=(2024, 3)).values.sum() == 2
        assert observations_mask(ds, month=(2024, 11)).values.sum() == 1
        assert observations_mask(ds, states=(1, 2), month=(2024, 5)).values.sum() == 2

    def test_time_dimension_is_reduced_to_two_dimensions(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)
        mask = observations_mask(ds, states=(4,))
        assert mask.dims == ("y", "x")
        assert mask.values.sum() == 2

    def test_mask_feeds_patches_from_mask(self):
        ds = observations_from_dated_codes(
            _projected_raster(_coded_grid()), stable_values=(STABLE,), fill_value=FILL
        )
        patches = patches_from_mask(observations_mask(ds, month=(2024, 3)))
        assert len(patches) == 1
        assert patches["pixel_count"].iloc[0] == 2
        assert patches["area_ha"].iloc[0] == pytest.approx(2 * 0.09)
        assert patches.crs.to_epsg() == 32633

    def test_alert_mask_feeds_patches_from_mask(self):
        alert, day = _alert_cube()
        ds = observations_from_annual_alert_days(alert, day)
        patches = patches_from_mask(observations_mask(ds, states=(4,)), connectivity=4)
        assert len(patches) == 2


class TestParquetRoundTrip:
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_round_trip_preserves_columns_dtypes_and_values(self, tmp_path, index):
        table = _all_three_tables()[index]
        path = tmp_path / f"observations_{index}.parquet"
        write_observations(table, path)
        restored = read_observations(path)

        assert list(restored.columns) == list(table.columns)
        assert [str(d) for d in restored.dtypes] == [str(d) for d in table.dtypes]
        assert len(restored) == len(table)
        assert restored["first_obs_date"].equals(table["first_obs_date"])
        assert restored["state"].equals(table["state"])
        assert restored["source"].astype(str).equals(table["source"].astype(str))

    def test_nat_survives_the_round_trip(self, tmp_path):
        coded = _all_three_tables()[0]
        assert coded["last_obs_date"].isna().all()
        path = tmp_path / "observations.parquet"
        write_observations(coded, path)
        assert read_observations(path)["last_obs_date"].isna().all()

    def test_refuses_a_table_missing_schema_columns(self, tmp_path):
        coded = _all_three_tables()[0].drop(columns=["confidence"])
        with pytest.raises(ValueError, match="missing observation column"):
            write_observations(coded, tmp_path / "bad.parquet")

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from ctreeskit import (
    calculate_categorical_area_stats,
    calculate_combined_categorical_area_stats,
    create_combined_classification,
)
from ctreeskit.xr_analyzer.xr_zonal_stats_module import (
    calculate_stats_with_categories,
)


def _time_class_raster():
    """Two-date 4x4 raster: rows 0-1 are class 1, rows 2-3 are class 2.

    Uses real WGS 84 coordinates over a patch of pantropical forest near the
    Amazon basin, with descending latitude (north-up), to exercise real
    coordinate handling.
    """
    y = np.array([10.00, 9.99, 9.98, 9.97])  # latitude, descending (north-up)
    x = np.array([-60.00, -59.99, -59.98, -59.97])  # longitude, ascending
    data = np.zeros((2, 4, 4))
    data[:, :2, :] = 1
    data[:, 2:, :] = 2
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": pd.date_range("2020-01-01", periods=2),
                "y": y, "x": x},
        name="classification",
    )


class TestCustomAreaDataArray(unittest.TestCase):
    """area_ds passed as a DataArray must work, not raise (issue #11)."""

    def setUp(self):
        self.classes = _time_class_raster()
        self.area = xr.DataArray(
            np.full((4, 4), 2.0),
            dims=["y", "x"],
            coords={"y": self.classes.y.values, "x": self.classes.x.values},
        )

    def test_dataarray_area(self):
        df = calculate_categorical_area_stats(self.classes, area_ds=self.area)
        # each class covers 8 pixels x 2.0 area units
        for t in range(2):
            self.assertAlmostEqual(df.iloc[t][1], 16.0)
            self.assertAlmostEqual(df.iloc[t][2], 16.0)
            self.assertAlmostEqual(df.iloc[t]["total_area"], 32.0)

    def test_none_area_counts_pixels(self):
        df = calculate_categorical_area_stats(self.classes, area_ds=None)
        for t in range(2):
            self.assertAlmostEqual(df.iloc[t][1], 8.0)
            self.assertAlmostEqual(df.iloc[t][2], 8.0)


class TestStatsWithCategories(unittest.TestCase):
    """calculate_stats_with_categories must return a well-formed DataFrame
    for both time and non-time inputs (issue #11)."""

    def setUp(self):
        # Real WGS 84 coordinates: descending latitude, ascending longitude.
        y = np.array([10.00, 9.99])
        x = np.array([-60.00, -59.99])
        self.coords = {"y": y, "x": x}
        self.categorical = xr.DataArray(
            [[1, 1], [2, 2]], dims=["y", "x"], coords=self.coords,
            name="classification")
        self.continuous = xr.DataArray(
            [[10.0, 20.0], [30.0, -5.0]], dims=["y", "x"], coords=self.coords,
            name="value")

    def _patch_match(self, continuous):
        return patch(
            "ctreeskit.xr_analyzer.xr_zonal_stats_module.reproject_match_ds",
            return_value=(continuous, None))

    def test_static_input(self):
        with self._patch_match(self.continuous):
            df = calculate_stats_with_categories(
                self.categorical, self.continuous)
        self.assertEqual(list(df["category"]), [1, 2])
        self.assertAlmostEqual(df.loc[df.category == 1, "mean_value"].item(), 15.0)
        self.assertAlmostEqual(df.loc[df.category == 1, "std_value"].item(), 5.0)
        # positive_only=True (default): -5.0 is excluded from class 2
        self.assertAlmostEqual(df.loc[df.category == 2, "mean_value"].item(), 30.0)

    def test_positive_only_false_keeps_nonpositive_values(self):
        with self._patch_match(self.continuous):
            df = calculate_stats_with_categories(
                self.categorical, self.continuous, positive_only=False)
        self.assertAlmostEqual(df.loc[df.category == 2, "mean_value"].item(), 12.5)

    def test_time_input(self):
        categorical_t = self.categorical.expand_dims(
            time=pd.date_range("2020-01-01", periods=2))
        with self._patch_match(self.continuous):
            df = calculate_stats_with_categories(
                categorical_t, self.continuous)
        # 2 time steps x 2 categories, with a time column on every row
        self.assertEqual(len(df), 4)
        self.assertIn("time", df.columns)
        self.assertFalse(df["time"].isna().any())
        self.assertEqual(list(df["category"]), [1, 2, 1, 2])

    def test_category_with_no_valid_values(self):
        continuous = xr.DataArray(
            [[10.0, 20.0], [-1.0, -5.0]], dims=["y", "x"], coords=self.coords,
            name="value")
        with self._patch_match(continuous):
            df = calculate_stats_with_categories(self.categorical, continuous)
        self.assertTrue(pd.isna(df.loc[df.category == 2, "mean_value"].item()))


def _static_class_raster():
    """10x10 static raster: rows 0-4 are class 1, rows 5-9 are class 2.

    Real WGS 84 coordinates with descending latitude (north-up).
    """
    y = np.linspace(10.00, 9.91, 10)
    x = np.linspace(-60.00, -59.91, 10)
    data = np.zeros((10, 10))
    data[:5, :] = 1
    data[5:, :] = 2
    return xr.DataArray(
        data, dims=["y", "x"], coords={"y": y, "x": x},
        name="classification",
    )


class TestStaticPathGoldenValues(unittest.TestCase):
    """Static (non-time) rasters must report every class (issue #10)."""

    def test_all_classes_reported(self):
        # 50 px of class 1 + 50 px of class 2; the pre-fix pivot kept only
        # class 1 and returned total_area=50
        df = calculate_categorical_area_stats(
            _static_class_raster(), area_ds=1.0)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0][1], 50.0)
        self.assertAlmostEqual(df.iloc[0][2], 50.0)
        self.assertAlmostEqual(df.iloc[0]["total_area"], 100.0)

    def test_static_and_time_paths_agree(self):
        static = _static_class_raster()
        with_time = static.expand_dims(
            time=pd.date_range("2020-01-01", periods=1))
        df_static = calculate_categorical_area_stats(static, area_ds=1.0)
        df_time = calculate_categorical_area_stats(with_time, area_ds=1.0)
        for col in (1, 2, "total_area"):
            self.assertAlmostEqual(
                df_static.iloc[0][col], df_time.iloc[0][col])

    def test_long_format_output(self):
        df = calculate_categorical_area_stats(
            _static_class_raster(), area_ds=1.0, reshape=False)
        self.assertEqual(
            list(df.columns), ["classification", "area_hectares"])
        by_class = df.set_index("classification")["area_hectares"]
        self.assertAlmostEqual(by_class[1], 50.0)
        self.assertAlmostEqual(by_class[2], 50.0)


class TestFlagValueLabels(unittest.TestCase):
    """Class labels must be matched by flag value, not position (issue #10)."""

    def _raster_with_flags(self):
        da = _static_class_raster()
        # non-contiguous land-cover-style codes
        da = da.where(da != 1, 10).where(da != 2, 30)
        da.attrs["flag_values"] = [10, 30]
        da.attrs["flag_meanings"] = "forest urban"
        return da

    def test_non_contiguous_codes_named_by_value(self):
        df = calculate_categorical_area_stats(
            self._raster_with_flags(), area_ds=1.0)
        self.assertAlmostEqual(df.iloc[0]["forest"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["urban"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["total_area"], 100.0)

    def test_meanings_only_fall_back_to_one_based_positions(self):
        da = _static_class_raster()  # classes 1 and 2
        da.attrs["flag_meanings"] = "forest urban"
        df = calculate_categorical_area_stats(da, area_ds=1.0)
        self.assertAlmostEqual(df.iloc[0]["forest"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["urban"], 50.0)

    def test_chunked_input_matches_numpy(self):
        da = self._raster_with_flags().chunk({"y": 5})
        df = calculate_categorical_area_stats(da, area_ds=1.0)
        self.assertAlmostEqual(df.iloc[0]["forest"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["urban"], 50.0)


class TestCombinedClassification(unittest.TestCase):
    """Combined class codes must be collision-free (issue #10)."""

    def _pair_rasters(self):
        base = _static_class_raster().rio.write_crs("EPSG:4326")
        # primary 3 over rows 0-4 and 4 over rows 5-9; secondary 12 and 2 on
        # the same split, so both (3, 12) and (4, 2) pairs occur — the float
        # encoding mapped both to 4.2
        primary = base.where(base != 1, 3).where(base != 2, 4)
        secondary = base.where(base != 1, 12).where(base != 2, 2)
        return primary, secondary

    def test_two_digit_class_codes_stay_distinct(self):
        primary, secondary = self._pair_rasters()
        combined = create_combined_classification(primary, secondary)
        self.assertEqual(combined.attrs["combined_modulus"], 100)
        self.assertEqual(set(np.unique(combined.values)), {312, 402})

    def test_combined_area_stats(self):
        primary, secondary = self._pair_rasters()
        df = calculate_combined_categorical_area_stats(
            primary, secondary, area_ds=1.0)
        self.assertAlmostEqual(df.iloc[0]["3 - 12"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["4 - 2"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["total_area"], 100.0)

    def test_combined_drop_zero(self):
        primary, secondary = self._pair_rasters()
        # zero out the secondary over the primary-3 rows: those pixels must
        # drop out of the combined accounting entirely
        secondary = secondary.where(primary != 3, 0)
        df = calculate_combined_categorical_area_stats(
            primary, secondary, area_ds=1.0)
        self.assertNotIn("3 - 12", df.columns)
        self.assertAlmostEqual(df.iloc[0]["4 - 2"], 50.0)
        self.assertAlmostEqual(df.iloc[0]["total_area"], 50.0)


if __name__ == "__main__":
    unittest.main()

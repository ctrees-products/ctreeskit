import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from ctreeskit import calculate_categorical_area_stats
from ctreeskit.xr_analyzer.xr_zonal_stats_module import (
    calculate_stats_with_categories,
)


def _time_class_raster():
    """Two-date 4x4 raster: rows 0-1 are class 1, rows 2-3 are class 2."""
    y = np.arange(4.0)
    x = np.arange(4.0)
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
        y = np.arange(2.0)
        x = np.arange(2.0)
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


if __name__ == "__main__":
    unittest.main()

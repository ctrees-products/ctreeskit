# import unittest
# import numpy as np
# import xarray as xr
# import pandas as pd
# from unittest.mock import patch, MagicMock

# from ctreeskit import (
#     calculate_categorical_area_stats
# )


# class TestZonalStats(unittest.TestCase):
#     def setUp(self):
#         """Set up test fixtures."""
#         # Create a simple 3x3 static categorical raster with classes 1-3
#         self.static_data = np.array([
#             [2, 3, 2],
#             [3, 1, 3],
#             [2, 3, 1]
#         ])
#         self.static_raster = xr.DataArray(
#             data=self.static_data,
#             dims=["y", "x"],
#             coords={
#                 "y": np.array([10, 20, 30]),
#                 "x": np.array([100, 200, 300])
#             }
#         )

#         # Create a simple 3x3 time-series categorical raster with 2 time slices
#         self.time_data = np.array([
#             # Time 0
#             [[2, 3, 2],
#              [3, 1, 3],
#              [2, 3, 1]],
#             # Time 1
#             [[2, 2, 2],
#              [2, 1, 3],
#              [2, 3, 3]]
#         ])
#         self.time_raster = xr.DataArray(
#             data=self.time_data,
#             dims=["time", "y", "x"],
#             coords={
#                 "time": pd.date_range("2020-01-01", periods=2),
#                 "y": np.array([10, 20, 30]),
#                 "x": np.array([100, 200, 300])
#             }
#         )

#         # Create a dataset version
#         self.dataset_raster = xr.Dataset(
#             {"classification": self.static_raster})

#         # Create an area DataArray
#         self.area_da = xr.DataArray(
#             data=np.ones((3, 3)) * 0.5,  # 0.5 ha per pixel
#             dims=["y", "x"],
#             coords={
#                 "y": np.array([10, 20, 30]),
#                 "x": np.array([100, 200, 300])
#             }
#         )

#     def test_static_raster_pixel_counting(self):
#         """Test basic pixel counting on static raster."""
#         result = calculate_categorical_area_stats(self.static_raster)
#         print(result)  # Debug: View the result structure

#         # With pixel counting, we expect:
#         # Total: 9 pixels (all classes included by default)
#         # Class 1: 2 pixels
#         # Class 2: 3 pixels
#         # Class 3: 4 pixels
#         self.assertEqual(result["total_area"][0], 9)
#         self.assertEqual(result[1][0], 2)
#         self.assertEqual(result[2][0], 3)
#         self.assertEqual(result[3][0], 4)

#     def test_static_raster_with_constant_area(self):
#         """Test using a constant area per pixel."""
#         result = calculate_categorical_area_stats(
#             self.static_raster, area_ds=2.0)
#         print(result)  # Debug: View the result structure

#         # With 2.0 ha per pixel, we expect:
#         # Total: 18 ha (9 pixels * 2 ha)
#         # Class 1: 4 ha (2 pixels * 2 ha)
#         # Class 2: 6 ha (3 pixels * 2 ha)
#         # Class 3: 8 ha (4 pixels * 2 ha)
#         self.assertEqual(result["total_area"][0], 18)
#         self.assertEqual(result[1][0], 4)
#         self.assertEqual(result[2][0], 6)
#         self.assertEqual(result[3][0], 8)

#     def test_static_raster_with_area_dataarray(self):
#         """Test using a DataArray for pixel areas."""
#         result = calculate_categorical_area_stats(
#             self.static_raster, area_ds=self.area_da)
#         print(result)  # Debug: View the result structure

#         # With 0.5 ha per pixel, we expect:
#         # Total: 4.5 ha (9 pixels * 0.5 ha)
#         # Class 1: 1.0 ha (2 pixels * 0.5 ha)
#         # Class 2: 1.5 ha (3 pixels * 0.5 ha)
#         # Class 3: 2.0 ha (4 pixels * 0.5 ha)
#         self.assertEqual(result["total_area"][0], 4.5)
#         self.assertEqual(result[1][0], 1.0)
#         self.assertEqual(result[2][0], 1.5)
#         self.assertEqual(result[3][0], 2.0)

#     @patch('ctreeskit.xr_analyzer.xr_zonal_stats_module.create_area_ds_from_degrees_ds')
#     def test_static_raster_with_calculated_area(self, mock_create_area):
#         """Test using True to calculate areas from coordinates."""
#         # Mock the area calculation function
#         mock_create_area.return_value = self.area_da

#         result = calculate_categorical_area_stats(
#             self.static_raster, area_ds=True)
#         print(result)  # Debug: View the result structure

#         # Same expectations as previous test
#         self.assertEqual(result["total_area"][0], 4.5)
#         self.assertEqual(result[1][0], 1.0)
#         self.assertEqual(result[2][0], 1.5)
#         self.assertEqual(result[3][0], 2.0)

#     def test_static_raster_with_drop_one(self):
#         """Test dropping a specific class - test with drop_one=True."""
#         # Create a modified version of the static raster with some zeros
#         data_with_zeros = np.array([
#             [0, 3, 0],
#             [3, 1, 3],
#             [2, 3, 1]
#         ])
#         raster_with_zeros = xr.DataArray(
#             data=data_with_zeros,
#             dims=["y", "x"],
#             coords=self.static_raster.coords
#         )

#         result = calculate_categorical_area_stats(
#             raster_with_zeros, drop_zero=True)
#         print(result)  # Debug: View the result structure

#         # With pixel counting and dropping zeros:
#         # Total: 7 pixels (9 total - 2 zeros)
#         # Class 1: 2 pixels
#         # Class 2: 1 pixels
#         # Class 3: 4 pixels
#         self.assertEqual(result["total_area"][0], 7)
#         self.assertEqual(result[1][0], 2)
#         self.assertEqual(result[2][0], 1)
#         self.assertEqual(result[3][0], 4)

#         # Confirm that zero class is not in the result
#         self.assertNotIn(0, result.columns)

#     def test_time_series_raster(self):
#         """Test processing a time-series raster."""
#         result = calculate_categorical_area_stats(self.time_raster)
#         print(result)  # Debug: View the result structure

#         # Should have 2 rows (one per time step)
#         # Time should be in the index, not columns
#         self.assertTrue(isinstance(result.index, pd.MultiIndex)
#                         or "time" in result.index.names)

#         # First time step:
#         # Total: 9 pixels
#         # Class 1: 2 pixels
#         # Class 2: 3 pixels
#         # Class 3: 4 pixels
#         self.assertEqual(result["total_area"][0], 9)
#         self.assertEqual(result[1][0], 2)
#         self.assertEqual(result[2][0], 3)
#         self.assertEqual(result[3][0], 4)

#         # Second time step:
#         # Class 1: 1 pixel
#         # Class 2: 5 pixels
#         # Class 3: 3 pixels
#         self.assertEqual(result["total_area"][1], 9)
#         self.assertEqual(result[1][1], 1)
#         self.assertEqual(result[2][1], 5)
#         self.assertEqual(result[3][1], 3)

#     def test_dataset_input(self):
#         """Test using a Dataset as input."""
#         result = calculate_categorical_area_stats(self.dataset_raster)
#         print(result)  # Debug: View the result structure

#         # With pixel counting, we expect:
#         # Total: 9 pixels
#         # Class 1: 2 pixels
#         # Class 2: 3 pixels
#         # Class 3: 4 pixels
#         self.assertEqual(result["total_area"][0], 9)
#         self.assertEqual(result['1'][0], 2)
#         self.assertEqual(result['2'][0], 3)
#         self.assertEqual(result['3'][0], 4)


# if __name__ == '__main__':
#     unittest.main()

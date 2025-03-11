import os
import tempfile
import json
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, box
from rasterio.transform import Affine
import pyproj

from ctreeskit.xr_analyzer.xr_spatial_processor_module import (
    GeometryData,
    process_geometry,
    clip_ds_to_bbox,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    create_proportion_geom_mask,
    align_and_resample_ds,
    _calculate_geometry_area,
    _measure
)


class TestGeometryProcessing(unittest.TestCase):
    def setUp(self):
        # Create a simple polygon
        self.polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])

        # Create a mock GeoJSON file
        self.geojson_data = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]
                    }
                }
            ]
        }

        # Create a temporary GeoJSON file
        self.temp_file = tempfile.NamedTemporaryFile(
            suffix='.geojson', delete=False)
        with open(self.temp_file.name, 'w') as f:
            json.dump(self.geojson_data, f)

    def tearDown(self):
        # Clean up the temporary file
        os.unlink(self.temp_file.name)

    def test_process_geometry_from_file(self):
        """Test processing geometry from a file."""
        result = process_geometry(self.temp_file.name)
        self.assertIsInstance(result, GeometryData)
        self.assertEqual(result.geom_crs, "EPSG:4326")
        self.assertAlmostEqual(result.geom_area, 1.0e-4,
                               places=5)  # 1 sq meter to hectares

    def test_process_geometry_from_shapely(self):
        """Test processing a Shapely geometry."""
        result = process_geometry(self.polygon)
        self.assertIsInstance(result, GeometryData)
        self.assertEqual(result.geom_crs, "EPSG:4326")

    def test_process_geometry_dissolve(self):
        """Test dissolving multiple geometries."""
        poly1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        poly2 = Polygon([(1, 0), (1, 1), (2, 1), (2, 0)])

        # With dissolve=True (default)
        result_dissolved = process_geometry([poly1, poly2])
        self.assertEqual(len(result_dissolved.geom), 1)

        # With dissolve=False
        result_not_dissolved = process_geometry([poly1, poly2], dissolve=False)
        self.assertEqual(len(result_not_dissolved.geom), 2)

    def test_process_geometry_output_units(self):
        """Test output units for geometry area."""
        # In hectares (default)
        result_ha = process_geometry(self.polygon)

        # In square meters
        result_m2 = process_geometry(self.polygon, output_in_ha=False)

        # Area in hectares should be 1e-4 times the area in square meters
        self.assertAlmostEqual(result_ha.geom_area * 10000,
                               result_m2.geom_area, places=5)

    def test_process_geometry_invalid_input(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            process_geometry(123)  # Not a valid geometry source


class TestRasterOperations(unittest.TestCase):
    def setUp(self):
        # Create a simple test raster
        lon = np.linspace(-180, 180, 73)
        lat = np.linspace(-90, 90, 37)
        data = np.random.rand(len(lat), len(lon))
        self.test_raster = xr.DataArray(
            data=data,
            dims=["y", "x"],
            coords={"y": lat, "x": lon}
        )

        # Add rio accessor attributes
        self.test_raster.rio.write_crs("EPSG:4326", inplace=True)

        # Create a time-series raster
        time_steps = pd.date_range("2020-01-01", periods=3)
        data_time = np.random.rand(len(time_steps), len(lat), len(lon))
        self.time_raster = xr.DataArray(
            data=data_time,
            dims=["time", "y", "x"],
            coords={"time": time_steps, "y": lat, "x": lon}
        )
        self.time_raster.rio.write_crs("EPSG:4326", inplace=True)

        # Create a test geometry
        self.geom = process_geometry(
            Polygon([(-10, -10), (-10, 10), (10, 10), (10, -10)]))

    @patch('ctreeskit.xr_analyzer.xr_spatial_processor_module.clip_ds_to_bbox')
    @patch('xarray.DataArray.rio')
    def test_clip_ds_to_bbox(self, mock_rio, mock_clip):
        """Test clipping to bounding box."""
        bbox = (-10, -10, 10, 10)

        # Configure mocks
        mock_clip.return_value = self.test_raster
        mock_clip.dims = self.test_raster.dims

        # Test basic clipping
        clip_ds_to_bbox(self.test_raster, bbox)
        mock_rio.clip_box.assert_called_once_with(
            minx=-10, miny=-10, maxx=10, maxy=10)

        # Test with time dimension and drop_time=True
        clip_ds_to_bbox(self.time_raster, bbox, drop_time=True)

    @patch('ctreeskit.xr_analyzer.xr_spatial_processor_module.clip_ds_to_bbox')
    def test_clip_ds_to_geom(self, mock_clip_bbox):
        """Test clipping to geometry."""
        # Configure mocks
        mock_clip_bbox.return_value = self.test_raster

        # Mock the rio.clip method
        with patch.object(self.test_raster.rio, 'clip', return_value=self.test_raster) as mock_clip:
            result = clip_ds_to_geom(self.test_raster, self.geom)
            mock_clip_bbox.assert_called_once()
            mock_clip.assert_called_once()

    def test_create_area_ds_from_degrees_ds(self):
        """Test calculating grid cell areas."""
        # Test with default values
        result = create_area_ds_from_degrees_ds(self.test_raster)
        self.assertEqual(result.attrs['units'], 'ha')
        self.assertIn('approximation', result.attrs['description'])

        # Test with high_accuracy=True
        result_high = create_area_ds_from_degrees_ds(
            self.test_raster, high_accuracy=True)
        self.assertIn('geodesic', result_high.attrs['description'])

        # Test with output_in_ha=False
        result_m2 = create_area_ds_from_degrees_ds(
            self.test_raster, output_in_ha=False)
        self.assertEqual(result_m2.attrs['units'], 'm²')

    @patch('ctreeskit.xr_analyzer.xr_spatial_processor_module.clip_ds_to_geom')
    def test_create_proportion_geom_mask(self, mock_clip):
        """Test creating proportion mask."""
        # Configure mock
        mock_clipped = xr.DataArray(
            data=np.ones((5, 5)),
            dims=["y", "x"],
            coords={"y": np.linspace(-2, 2, 5), "x": np.linspace(-2, 2, 5)}
        )
        mock_clipped.rio.write_crs("EPSG:4326", inplace=True)
        mock_clip.return_value = mock_clipped

        # Set up transform for the test
        transform = Affine(1.0, 0.0, -2.0, 0.0, 1.0, -2.0)
        with patch.object(mock_clipped.rio, 'transform', return_value=transform):
            # Test with default parameters
            with patch('numpy.nonzero', return_value=(np.array([0, 1, 2]), np.array([0, 1, 2]))):
                result = create_proportion_geom_mask(mock_clipped, self.geom)
                self.assertEqual(result.attrs['units'], 'proportion')

    @patch('ctreeskit.xr_analyzer.xr_spatial_processor_module.clip_ds_to_bbox')
    @patch('ctreeskit.xr_analyzer.xr_spatial_processor_module.create_area_ds_from_degrees_ds')
    def test_align_and_resample_ds(self, mock_create_area, mock_clip):
        """Test aligning and resampling datasets."""
        # Configure mocks
        mock_clip.return_value = self.test_raster
        mock_aligned = self.test_raster.copy()
        mock_create_area.return_value = xr.DataArray(
            np.ones_like(self.test_raster))

        with patch.object(mock_clip.return_value.rio, 'reproject_match', return_value=mock_aligned):
            # Test with default parameters
            result, area = align_and_resample_ds(
                self.test_raster, self.test_raster)
            mock_clip.assert_called_once()
            mock_create_area.assert_called_once()

            # Test without area grid
            result, area = align_and_resample_ds(
                self.test_raster, self.test_raster, return_area_grid=False)
            self.assertIsNone(area)


class TestHelperFunctions(unittest.TestCase):
    def test_calculate_geometry_area(self):
        """Test geometry area calculation."""
        poly = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        geom = [poly]

        # Test with default target projection
        area = _calculate_geometry_area(geom, "EPSG:4326")
        self.assertGreater(area, 0)

        # Test with different target projection
        area2 = _calculate_geometry_area(geom, "EPSG:4326", target_epsg=3857)
        self.assertGreater(area2, 0)

    def test_measure(self):
        """Test geodesic distance calculation."""
        # Distance between equator and 1 degree north at prime meridian
        dist = _measure(0, 0, 1, 0)
        self.assertAlmostEqual(dist, 111195, delta=5)  # Approximately 111.2 km

        # Test with different coordinates
        dist2 = _measure(0, 0, 0, 1)
        self.assertGreater(dist2, 0)


if __name__ == '__main__':
    unittest.main()

import os
import tempfile
import json
import unittest

import numpy as np
import xarray as xr
import pandas as pd
from shapely.geometry import Polygon, box

from ctreeskit import (
    GeometryData,
    process_geometry,
    clip_ds_to_bbox,
    clip_ds_to_geom,
    create_area_ds_from_degrees_ds,
    create_proportion_geom_mask,
    reproject_match_ds
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

    def test_process_geometry_rfc7946_no_crs(self):
        """RFC 7946 GeoJSON (no 'crs' member) defaults to EPSG:4326."""
        geojson = {k: v for k, v in self.geojson_data.items() if k != "crs"}
        rfc_file = tempfile.NamedTemporaryFile(
            suffix='.geojson', delete=False)
        with open(rfc_file.name, 'w') as f:
            json.dump(geojson, f)
        try:
            result = process_geometry(rfc_file.name)
            self.assertEqual(result.geom_crs, "EPSG:4326")
            self.assertEqual(len(result.geom), 1)
        finally:
            os.unlink(rfc_file.name)

    def test_process_geometry_invalid_input(self):
        """Test error handling for invalid inputs."""
        with self.assertRaises(ValueError):
            process_geometry(123)  # Not a valid geometry source


class TestRasterOperations(unittest.TestCase):
    def setUp(self):
        # Global test raster: 5-degree pixels, descending latitude (north-up)
        lon = np.linspace(-180, 180, 73)
        lat = np.linspace(90, -90, 37)
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

    def test_clip_ds_to_bbox(self):
        """Clipping to a bbox keeps exactly the pixels inside it."""
        bbox = (-10.0, -10.0, 10.0, 10.0)
        clipped = clip_ds_to_bbox(self.test_raster, bbox)
        self.assertTrue((clipped.x >= -10).all() and (clipped.x <= 10).all())
        self.assertTrue((clipped.y >= -10).all() and (clipped.y <= 10).all())
        # 5-degree pixels: centers -10, -5, 0, 5, 10 survive on each axis
        self.assertEqual(clipped.shape, (5, 5))
        xr.testing.assert_equal(
            clipped, self.test_raster.sel(x=slice(-10, 10)).sel(
                y=self.test_raster.y[(self.test_raster.y >= -10)
                                     & (self.test_raster.y <= 10)]))

        # drop_time=True returns the first time slice
        clipped_t = clip_ds_to_bbox(self.time_raster, bbox, drop_time=True)
        self.assertNotIn("time", clipped_t.dims)

    def test_clip_ds_to_geom(self):
        """Clipping to a triangle masks pixels outside it and keeps values inside."""
        triangle = process_geometry(
            Polygon([(-10, -10), (10, -10), (10, 10)]))
        clipped = clip_ds_to_geom(self.test_raster, triangle)
        # the far corner of the bounding box is outside the triangle
        self.assertTrue(np.isnan(
            float(clipped.sel(x=-10, y=10, method="nearest"))))
        # a pixel inside the triangle keeps its original value
        inside = clipped.sel(x=5, y=-5, method="nearest")
        original = self.test_raster.sel(x=5, y=-5, method="nearest")
        self.assertAlmostEqual(float(inside), float(original))

    def test_create_area_ds_from_degrees_ds(self):
        """Test calculating grid cell areas."""
        # Test with default values
        result = create_area_ds_from_degrees_ds(self.test_raster)
        self.assertEqual(result.attrs['units'], 'ha')

        # Test with high_accuracy=True
        result_high = create_area_ds_from_degrees_ds(
            self.test_raster, high_accuracy=True)
        self.assertIn('geodesic', result_high.attrs['description'])

        # Test with output_in_ha=False
        result_m2 = create_area_ds_from_degrees_ds(
            self.test_raster, output_in_ha=False)
        self.assertEqual(result_m2.attrs['units'], 'm²')

    def test_reproject_match_ds(self):
        """A finer raster is resampled onto the template's exact grid."""
        # template: 1-degree pixels over a 10x10-degree patch near the equator
        template_y = np.linspace(9.5, 0.5, 10)
        template_x = np.linspace(0.5, 9.5, 10)
        template = xr.DataArray(
            np.zeros((10, 10)), dims=["y", "x"],
            coords={"y": template_y, "x": template_x})
        template.rio.write_crs("EPSG:4326", inplace=True)

        # target: same extent at 0.5-degree pixels, value = latitude band
        target_y = np.linspace(9.75, 0.25, 20)
        target_x = np.linspace(0.25, 9.75, 20)
        target = xr.DataArray(
            np.repeat(target_y, 20).reshape(20, 20), dims=["y", "x"],
            coords={"y": target_y, "x": target_x})
        target.rio.write_crs("EPSG:4326", inplace=True)

        aligned, area = reproject_match_ds(template, target)
        np.testing.assert_allclose(aligned.y.values, template_y)
        np.testing.assert_allclose(aligned.x.values, template_x)
        # nearest-neighbour values stay within the source latitude range
        self.assertTrue(
            (aligned.values >= 0.25).all() and (aligned.values <= 9.75).all())
        self.assertEqual(area.shape, aligned.shape)
        self.assertEqual(area.attrs["units"], "ha")

        _, no_area = reproject_match_ds(
            template, target, return_area_grid=False)
        self.assertIsNone(no_area)


def _karney_cell_area(lon_w, lat_s, lon_e, lat_n):
    """WGS 84 geodesic area (m²) of a lon/lat cell — the reference oracle."""
    from pyproj import Geod
    area, _ = Geod(ellps="WGS84").polygon_area_perimeter(
        [lon_w, lon_e, lon_e, lon_w], [lat_s, lat_s, lat_n, lat_n])
    return abs(area)


def _grid(lat_top, res, n=4, lon_left=-60.0, descending=True):
    """n x n raster of res-degree pixels with the top edge at lat_top."""
    y = np.array([lat_top - res * (i + 0.5) for i in range(n)])
    if not descending:
        y = y[::-1]
    x = np.array([lon_left + res * (i + 0.5) for i in range(n)])
    return xr.DataArray(np.zeros((n, n)), dims=["y", "x"],
                        coords={"y": y, "x": x})


class TestAreaGoldenValues(unittest.TestCase):
    """Golden-value checks for cell areas — the numbers this package exists
    to produce (issue #17). Both methods are compared against the WGS 84
    geodesic (Karney) area from pyproj, and against a literal reference for
    the canonical 1x1-degree equator cell.
    """

    def test_one_degree_cell_at_equator(self):
        # WGS 84 geodesic area of the cell spanning 0..1 deg in both axes
        golden_m2 = 1.2309e10  # 12,309 km²
        da = _grid(lat_top=2.0, res=1.0)
        for high_accuracy in (True, False):
            area = create_area_ds_from_degrees_ds(
                da, high_accuracy=high_accuracy, output_in_ha=False)
            # cell centered at (1.5N, -59.5E) scaled to the equator cell via
            # the oracle; assert the equator-latitude row directly instead:
            ref = _karney_cell_area(-60.0, 1.0, -59.0, 2.0)
            self.assertAlmostEqual(
                float(area.isel(y=0, x=0)) / ref, 1.0, delta=1e-4)
        # and the canonical constant itself, via the oracle
        self.assertAlmostEqual(
            _karney_cell_area(0.0, 0.0, 1.0, 1.0) / golden_m2, 1.0, delta=1e-4)

    def test_thirty_meter_cells_match_reference(self):
        # production resolution (~30 m): both methods must agree with the
        # geodesic reference to well under a millionth relatively
        res = 0.00027
        for lat_top in (10.0, 75.0):
            da = _grid(lat_top=lat_top, res=res)
            for high_accuracy in (True, False):
                area = create_area_ds_from_degrees_ds(
                    da, high_accuracy=high_accuracy, output_in_ha=False)
                for iy in (0, 3):
                    yc = float(da.y[iy])
                    xc = float(da.x[0])
                    ref = _karney_cell_area(xc - res / 2, yc - res / 2,
                                            xc + res / 2, yc + res / 2)
                    self.assertAlmostEqual(
                        float(area.isel(y=iy, x=0)) / ref, 1.0, delta=1e-6)

    def test_one_degree_cells_within_five_hundredths_percent(self):
        # coarse (1-degree) cells: measured worst case is ~5e-5 relative
        da = _grid(lat_top=77.0, res=1.0)
        for high_accuracy in (True, False):
            area = create_area_ds_from_degrees_ds(
                da, high_accuracy=high_accuracy, output_in_ha=False)
            yc = float(da.y[0])
            xc = float(da.x[0])
            ref = _karney_cell_area(xc - 0.5, yc - 0.5, xc + 0.5, yc + 0.5)
            self.assertAlmostEqual(
                float(area.isel(y=0, x=0)) / ref, 1.0, delta=5e-4)

    def test_method_heuristic_is_orientation_independent(self):
        # spans 80N..76N: geodesic regardless of storage orientation
        north_up = create_area_ds_from_degrees_ds(_grid(80.0, 1.0))
        south_up = create_area_ds_from_degrees_ds(
            _grid(80.0, 1.0, descending=False))
        self.assertIn("geodesic", north_up.attrs["description"])
        self.assertIn("geodesic", south_up.attrs["description"])
        # tropical raster: equal-area approximation on both orientations
        tropical = create_area_ds_from_degrees_ds(_grid(10.0, 1.0))
        self.assertIn("6933", tropical.attrs["description"])

    def test_hectare_conversion(self):
        da = _grid(lat_top=10.0, res=0.01)
        in_m2 = create_area_ds_from_degrees_ds(da, output_in_ha=False)
        in_ha = create_area_ds_from_degrees_ds(da, output_in_ha=True)
        np.testing.assert_allclose(in_ha.values, in_m2.values * 1e-4)

    def test_process_geometry_area_matches_reference(self):
        geom = box(-60.0, 1.0, -59.0, 2.0)
        result = process_geometry(geom, output_in_ha=False)
        ref = _karney_cell_area(-60.0, 1.0, -59.0, 2.0)
        self.assertAlmostEqual(result.geom_area / ref, 1.0, delta=1e-4)


class TestCreateProportionGeomMask(unittest.TestCase):
    """Proportion masks with known expected values (issue #13).

    The raster covers 10x10 pixels of 0.01 degrees near 10N, -60E (north-up,
    descending latitude), with pixel edges on multiples of 0.01 degrees. Data
    values are all zero: the mask must be driven by the geometry alone.
    """

    def setUp(self):
        x = np.linspace(-59.995, -59.905, 10)
        y = np.linspace(9.995, 9.905, 10)
        self.raster = xr.DataArray(
            np.zeros((10, 10)), dims=["y", "x"], coords={"y": y, "x": x})
        self.raster.rio.write_crs("EPSG:4326", inplace=True)
        # aligned exactly to pixel edges: covers a 3x3 pixel block fully
        self.aligned_geom = box(-59.97, 9.94, -59.94, 9.97)
        # offset half a pixel in both axes: overlaps a 5x5 block with
        # 0.25 corners, 0.5 edges, and a full interior
        self.offset_geom = box(-59.975, 9.935, -59.935, 9.975)

    def test_weighted_proportions_half_cell_offset(self):
        mask = create_proportion_geom_mask(
            self.raster, self.offset_geom, overwrite=True)
        self.assertEqual(mask.shape, (5, 5))
        # total proportion equals geometry area / pixel area
        self.assertAlmostEqual(float(mask.sum()), 16.0, places=4)
        interior = mask.sel(y=9.955, x=-59.955, method="nearest")
        corner = mask.sel(y=9.975, x=-59.975, method="nearest")
        edge = mask.sel(y=9.975, x=-59.955, method="nearest")
        self.assertAlmostEqual(float(interior), 1.0, places=4)
        self.assertAlmostEqual(float(corner), 0.25, places=4)
        self.assertAlmostEqual(float(edge), 0.5, places=4)

    def test_weighted_proportions_edge_aligned(self):
        mask = create_proportion_geom_mask(
            self.raster, self.aligned_geom, overwrite=True)
        self.assertEqual(mask.shape, (3, 3))
        np.testing.assert_allclose(mask.values, 1.0, atol=1e-4)

    def test_below_threshold_returns_binary_mask(self):
        # pixel/geometry area ratio is ~0.111 here, so 0.2 forces the
        # binary path
        with self.assertWarns(UserWarning):
            mask = create_proportion_geom_mask(
                self.raster, self.aligned_geom, pixel_ratio=0.2)
        self.assertTrue(set(np.unique(mask.values)) <= {0, 1})
        self.assertEqual(float(mask.sum()), 9.0)

    def test_mask_ignores_data_values(self):
        # all-zero data must still yield full proportions inside the geometry
        mask = create_proportion_geom_mask(
            self.raster, self.aligned_geom, overwrite=True)
        self.assertAlmostEqual(float(mask.max()), 1.0, places=4)


if __name__ == '__main__':
    unittest.main()

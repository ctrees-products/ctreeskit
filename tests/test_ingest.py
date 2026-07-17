import tempfile
import unittest

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)

from ctreeskit.arraylake_tools.ingest import AnnualRasterIngester


class TestIndexSlice(unittest.TestCase):
    """_index_slice must locate aligned windows and reject off-grid ones
    instead of silently snapping to the nearest index (issue #16)."""

    def setUp(self):
        self.stored = np.arange(0.0, 100.0, 1.0)

    def test_aligned_subset_round_trips(self):
        subset = self.stored[10:20]
        s = AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertEqual(s, slice(10, 20))
        np.testing.assert_array_equal(self.stored[s], subset)

    def test_float_drift_within_tolerance_is_accepted(self):
        subset = self.stored[10:20] + 1e-9
        s = AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertEqual(s, slice(10, 20))

    def test_off_grid_subset_raises(self):
        subset = self.stored[10:20] + 0.4  # off-origin by 0.4 of a cell
        with self.assertRaises(ValueError) as ctx:
            AnnualRasterIngester._index_slice(self.stored, subset)
        self.assertIn("align", str(ctx.exception))

    def test_different_resolution_raises(self):
        subset = np.arange(10.0, 20.0, 0.5)
        with self.assertRaises(ValueError):
            AnnualRasterIngester._index_slice(self.stored, subset)

    def test_subset_extending_beyond_grid_raises(self):
        subset = np.arange(95.0, 105.0, 1.0)
        with self.assertRaises(ValueError):
            AnnualRasterIngester._index_slice(self.stored, subset)

    def test_descending_coords_aligned(self):
        stored_desc = self.stored[::-1].copy()
        subset = stored_desc[10:20]
        s = AnnualRasterIngester._index_slice(stored_desc, subset)
        self.assertEqual(s, slice(10, 20))


class _LocalRepoClient:
    """Client stand-in that resolves every repo name to a local Icechunk repo."""

    def __init__(self, repo: icechunk.Repository):
        self._repo = repo

    def get_repo(self, name: str) -> icechunk.Repository:
        return self._repo


class TestAnnualRasterIngesterEndToEnd(unittest.TestCase):
    """Full GeoTIFF -> Icechunk ingestion against a real local repository.

    Annual mosaics are real GeoTIFFs on a WGS 84 grid of 0.01-degree pixels
    near 10N, -60E (north-up, descending latitude); the repo is a real
    ``icechunk.Repository`` on local-filesystem storage, so every write and
    read-back exercises the production code path end to end.
    """

    NODATA = -1

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = self._tmp.name

        self.x = np.linspace(-59.995, -59.805, 20)
        self.y = np.linspace(9.995, 9.805, 20)  # descending (north-up)
        for year in (2019, 2020, 2021, 2022):
            self._write_mosaic(f"{tmp}/mosaic_{year}.tif", year)
        # a mosaic whose grid origin is shifted by half a pixel
        self._write_mosaic(f"{tmp}/shifted_2021.tif", 2021,
                           x=self.x + 0.005, y=self.y + 0.005)

        self.config = {
            "dataset_name": "annual-test",
            "organization": "testorg",
            "crs": "EPSG:4326",
            "chunks": {"time": 1, "y": 10, "x": 10},
            "groups": {
                "annual": {
                    "time": {"start": "2020-01-01", "end": "2021-12-31",
                             "freq": "YS"},
                    "classification": {
                        "unit_type": "int",
                        "nodata": self.NODATA,
                        "s3_path_prefix": f"{tmp}/mosaic_",
                        "s3_path_suffix": ".tif",
                        "values": {"1": "stable forest", "2": "forest loss"},
                    },
                },
            },
        }
        repo = icechunk.Repository.create(
            icechunk.local_filesystem_storage(f"{tmp}/repo"))
        self.ingester = AnnualRasterIngester(
            self.config, client=_LocalRepoClient(repo))
        self.repo = repo

    def tearDown(self):
        self._tmp.cleanup()

    def _write_mosaic(self, path, year, x=None, y=None):
        """GeoTIFF filled with ``year - 2000``, nodata in the 2x2 NW corner."""
        data = np.full((20, 20), year - 2000, dtype="int16")
        data[:2, :2] = self.NODATA
        da = xr.DataArray(
            data, dims=["y", "x"],
            coords={"y": y if y is not None else self.y,
                    "x": x if x is not None else self.x},
        )
        da = da.rio.write_crs("EPSG:4326").rio.write_nodata(self.NODATA)
        da.rio.to_raster(path)

    def _stored(self):
        return xr.open_zarr(
            self.repo.readonly_session("main").store, group="annual",
            consolidated=False, chunks=None,
        )

    def test_initialize_schema_allocates_empty_domain(self):
        self.ingester.initialize_schema()
        stored = self._stored()
        self.assertEqual(stored["classification"].shape, (2, 20, 20))
        self.assertEqual(
            pd.to_datetime(stored.time.values).year.tolist(), [2020, 2021])
        np.testing.assert_array_equal(stored.y.values, self.y)
        np.testing.assert_array_equal(stored.x.values, self.x)
        # no data chunks were written: everything reads back as the fill value
        self.assertTrue((stored["classification"].values == self.NODATA).all())
        # CF flag metadata from the config's "values" mapping
        attrs = stored["classification"].attrs
        np.testing.assert_array_equal(attrs["flag_values"], [1, 2])
        self.assertEqual(attrs["flag_meanings"], "stable_forest forest_loss")

    def test_region_write_fills_only_target_year(self):
        self.ingester.initialize_schema()
        self.ingester.ingest_year(2020)
        stored = self._stored()["classification"]
        year0 = stored.isel(time=0).values
        self.assertTrue((year0[2:, 2:] == 20).all())
        self.assertTrue((year0[:2, :2] == self.NODATA).all())
        # 2021's slot is untouched
        self.assertTrue(
            (stored.isel(time=1).values == self.NODATA).all())

    def test_bbox_subset_writes_only_window(self):
        self.ingester.initialize_schema()
        # 6x6 window on pixel centers 5..10; the epsilon absorbs the
        # float noise of coordinates regenerated from the GeoTIFF transform
        eps = 1e-6
        bbox = (self.x[5] - eps, self.y[10] - eps,
                self.x[10] + eps, self.y[5] + eps)
        self.ingester.ingest_year(2021, bbox=bbox)
        stored = self._stored()["classification"]
        year1 = stored.isel(time=1)
        window = year1.sel(x=slice(bbox[0], bbox[2]),
                           y=slice(bbox[3], bbox[1]))
        self.assertEqual(window.shape, (6, 6))
        self.assertTrue((window.values == 21).all())
        # everything outside the window stays nodata
        self.assertEqual(int((year1.values == 21).sum()), 36)

    def test_shifted_grid_bbox_is_rejected(self):
        self.ingester.initialize_schema()
        self.ingester.s3_path_prefix = f"{self._tmp.name}/shifted_"
        with self.assertRaises(ValueError) as ctx:
            self.ingester.ingest_year(
                2021, bbox=(-59.945, 9.895, -59.895, 9.945))
        self.assertIn("align", str(ctx.exception))

    def test_append_extends_time_axis(self):
        self.ingester.initialize_schema()
        self.ingester.ingest_year(2020)
        self.ingester.ingest_year(2021)
        self.ingester.ingest_year(2022)
        stored = self._stored()["classification"]
        self.assertEqual(stored.sizes["time"], 3)
        self.assertTrue((stored.isel(time=2).values[2:, 2:] == 22).all())

    def test_append_with_bbox_is_rejected(self):
        self.ingester.initialize_schema()
        with self.assertRaises(ValueError):
            self.ingester.ingest_year(
                2022, bbox=(-59.945, 9.895, -59.895, 9.945))

    def test_year_before_axis_is_rejected(self):
        self.ingester.initialize_schema()
        with self.assertRaises(ValueError):
            self.ingester.ingest_year(2019)

    def test_tag_points_at_ingest_snapshot(self):
        self.ingester.initialize_schema()
        snapshot = self.ingester.ingest_year(2020, tag="2020-release")
        self.assertIn("2020-release", self.repo.list_tags())
        self.assertEqual(self.repo.lookup_tag("2020-release"), snapshot)


if __name__ == "__main__":
    unittest.main()

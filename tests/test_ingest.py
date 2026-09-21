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
        # a mosaic one chunk (10 cells = 0.1 deg) further north, on the same lattice
        self._write_mosaic(f"{tmp}/north_2020.tif", 2020, y=self.y + 0.1)

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

    # ---------------------------------------------------------- coordinate placement

    def test_full_extent_source_with_shifted_origin_is_rejected(self):
        """Same shape as the grid but a different origin must raise, not misalign."""
        self.ingester.initialize_schema()
        self.ingester.s3_path_prefix = f"{self._tmp.name}/shifted_"
        with self.assertRaises(ValueError) as ctx:
            self.ingester.ingest_year(2021)
        self.assertIn("align", str(ctx.exception))
        # nothing was written
        self.assertTrue(
            (self._stored()["classification"].values == self.NODATA).all())

    def test_source_larger_than_grid_is_rejected(self):
        self.ingester.initialize_schema()
        self.ingester.s3_path_prefix = f"{self._tmp.name}/north_"
        with self.assertRaises(ValueError) as ctx:
            self.ingester.ingest_year(2020)
        self.assertIn("grow_extent", str(ctx.exception))

    def test_extent_config_defines_grid_and_source_fills_window(self):
        """A configured extent larger than the template: grid comes from the
        extent, and a full mosaic lands in its own window by coordinate."""
        # 40x40 domain: one chunk further north and one further east
        self.config["groups"]["annual"]["extent"] = [-60.0, 9.8, -59.6, 10.2]
        ingester = AnnualRasterIngester(self.config, client=_LocalRepoClient(self.repo))
        ingester.initialize_schema()
        stored = self._stored()
        self.assertEqual(stored["classification"].shape, (2, 40, 40))
        self.assertAlmostEqual(float(stored.y.values[0]), 10.195)
        self.assertAlmostEqual(float(stored.x.values[-1]), -59.605)
        ingester.ingest_year(2020)
        year0 = self._stored()["classification"].isel(time=0)
        # mosaic occupies rows 20..39 (south half), cols 0..19 (west half)
        block = year0.values[20:, :20]
        self.assertTrue((block[2:, 2:] == 20).all())
        self.assertEqual(int((year0.values == 20).sum()), 20 * 20 - 4)

    def test_extent_off_lattice_template_is_rejected(self):
        self.config["groups"]["annual"]["extent"] = [-60.005, 9.8, -59.6, 10.2]
        ingester = AnnualRasterIngester(self.config, client=_LocalRepoClient(self.repo))
        with self.assertRaises(ValueError):
            ingester.initialize_schema()

    # ---------------------------------------------------------- group selection

    def test_multi_group_config_requires_group(self):
        cfg = dict(self.config)
        cfg["groups"] = {"annual": self.config["groups"]["annual"],
                         "other": self.config["groups"]["annual"]}
        with self.assertRaises(ValueError) as ctx:
            AnnualRasterIngester(cfg, client=_LocalRepoClient(self.repo))
        self.assertIn("group=", str(ctx.exception))
        ing = AnnualRasterIngester(cfg, client=_LocalRepoClient(self.repo),
                                   group="other")
        self.assertEqual(ing.group_name, "other")
        with self.assertRaises(ValueError):
            AnnualRasterIngester(cfg, client=_LocalRepoClient(self.repo),
                                 group="missing")

    # ---------------------------------------------------------- extent growth

    def test_grow_extent_north_preserves_data_and_accepts_new_source(self):
        self.ingester.initialize_schema()
        self.ingester.ingest_year(2020)
        before = self._stored()["classification"].isel(time=0).values.copy()
        # grow one chunk (10 cells) north: top edge 10.0 -> 10.1
        self.ingester.grow_extent((-60.0, 9.8, -59.8, 10.1), verify_samples=20)
        stored = self._stored()
        arr = stored["classification"]
        self.assertEqual(arr.shape, (2, 30, 20))
        self.assertAlmostEqual(float(stored.y.values[0]), 10.095)
        self.assertAlmostEqual(float(stored.y.values[-1]), 9.805)
        np.testing.assert_allclose(stored.x.values, self.x, atol=1e-9)
        after = arr.isel(time=0).values
        # old data moved down by 10 rows, byte for byte; new rows are nodata
        np.testing.assert_array_equal(after[10:], before)
        self.assertTrue((after[:10] == self.NODATA).all())
        # 2021 slot still empty
        self.assertTrue((arr.isel(time=1).values == self.NODATA).all())
        # a mosaic covering the new northern rows (and overlapping 10 old rows)
        # now fits and is placed by coordinate
        self.ingester.s3_path_prefix = f"{self._tmp.name}/north_"
        self.ingester.ingest_year(2020)
        after2 = self._stored()["classification"].isel(time=0).values
        self.assertTrue((after2[:20][2:, 2:] == 20).all())
        self.assertTrue((after2[:2, :2] == self.NODATA).all())
        np.testing.assert_array_equal(after2[20:], before[10:])
        # history: the pre-growth snapshot is still there
        msgs = [c.message for c in self.repo.ancestry(branch="main")]
        self.assertTrue(any(m.startswith("grow extent annual/classification") for m in msgs))

    def test_grow_extent_defaults_to_configured_extent(self):
        """With no argument, the domain converges on the config's extent."""
        self.ingester.initialize_schema()
        with self.assertRaises(ValueError):
            self.ingester.grow_extent(verify_samples=0)  # config has no extent
        self.ingester.extent = [-60.0, 9.8, -59.6, 10.2]
        self.ingester.grow_extent(verify_samples=0)
        stored = self._stored()
        self.assertEqual(stored["classification"].shape, (2, 40, 40))
        self.assertAlmostEqual(float(stored.y.values[0]), 10.195)
        self.assertAlmostEqual(float(stored.x.values[-1]), -59.605)

    def test_grow_extent_snaps_low_index_growth_to_chunks(self):
        self.ingester.initialize_schema()
        # 5 rows north is half a chunk: snapped outward to 10
        self.ingester.grow_extent((-60.0, 9.8, -59.8, 10.05), verify_samples=0)
        self.assertEqual(self._stored()["classification"].shape, (2, 30, 20))
        with self.assertRaises(ValueError):
            self.ingester.grow_extent((-60.0, 9.8, -59.8, 10.15),
                                      snap_to_chunks=False, verify_samples=0)

    def test_grow_extent_east_and_south_need_no_chunk_alignment(self):
        self.ingester.initialize_schema()
        self.ingester.ingest_year(2020)
        # 3 cells east, 7 cells south (high-index sides: plain resize)
        self.ingester.grow_extent((-60.0, 9.73, -59.77, 10.0), verify_samples=10)
        stored = self._stored()
        arr = stored["classification"]
        self.assertEqual(arr.shape, (2, 27, 23))
        self.assertTrue((arr.isel(time=0).values[:20, :20][2:, 2:] == 20).all())
        self.assertTrue((arr.isel(time=0).values[20:, :] == self.NODATA).all())
        self.assertTrue((arr.isel(time=0).values[:, 20:] == self.NODATA).all())

    def test_grow_extent_tolerates_float_noise_on_unchanged_edges(self):
        self.ingester.initialize_schema()
        # right/bottom edges given with sub-cell float noise must not count as shrink
        self.ingester.grow_extent((-60.0, 9.8 + 3e-13, -59.8 - 6e-13, 10.1),
                                  verify_samples=0)
        self.assertEqual(self._stored()["classification"].shape, (2, 30, 20))

    def test_grow_extent_rejects_shrink_and_off_lattice(self):
        self.ingester.initialize_schema()
        with self.assertRaises(ValueError):
            self.ingester.grow_extent((-60.0, 9.9, -59.8, 10.0), verify_samples=0)
        with self.assertRaises(ValueError):
            self.ingester.grow_extent((-60.0, 9.8, -59.8, 10.103), verify_samples=0)
        with self.assertRaises(ValueError):
            self.ingester.grow_extent((-60.0, 9.8, -59.8, 10.0), verify_samples=0)


if __name__ == "__main__":
    unittest.main()

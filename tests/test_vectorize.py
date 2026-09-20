"""Tests for xr_vectorize_module, over small synthetic rasters.

Grids use realistic WGS 84 coordinates (descending latitude, north-up) or a
metre-based projected CRS, so the area code paths are exercised for real.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("geopandas")
pytest.importorskip("pyarrow")

import geopandas as gpd  # noqa: E402
from shapely.geometry import box  # noqa: E402

from ctreeskit import (  # noqa: E402
    assign_to_polygons,
    merge_patches,
    patches_from_categorical,
    patches_from_mask,
    patches_over_time,
    read_geoparquet,
    write_geoparquet,
)

# ~5 m at the equator (one degree of latitude is about 111 km).
DEG_5M = 4.5e-5


def _geographic_raster(values, res=DEG_5M, lon0=-60.0, lat0=0.0):
    """Wrap a 2-D array on a north-up WGS 84 grid at the given resolution."""
    values = np.asarray(values)
    ny, nx = values.shape
    x = lon0 + (np.arange(nx) + 0.5) * res
    y = lat0 - (np.arange(ny) + 0.5) * res
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])
    return da.rio.write_crs("EPSG:4326")


def _projected_raster(values, res=30.0, x0=500000.0, y0=4000000.0, crs="EPSG:32633"):
    """Wrap a 2-D array on a north-up metre-based grid at the given resolution."""
    values = np.asarray(values)
    ny, nx = values.shape
    x = x0 + (np.arange(nx) + 0.5) * res
    y = y0 - (np.arange(ny) + 0.5) * res
    da = xr.DataArray(values, coords={"y": y, "x": x}, dims=["y", "x"])
    return da.rio.write_crs(crs)


class TestConnectivity:
    def test_diagonal_blobs_merge_under_connectivity_8(self):
        grid = np.zeros((6, 6), dtype="uint8")
        grid[1:3, 1:3] = 1
        grid[3:5, 3:5] = 1  # touches the first block corner to corner
        patches = patches_from_mask(_projected_raster(grid), connectivity=8)
        assert len(patches) == 1
        assert patches["pixel_count"].sum() == 8

    def test_diagonal_blobs_split_under_connectivity_4(self):
        grid = np.zeros((6, 6), dtype="uint8")
        grid[1:3, 1:3] = 1
        grid[3:5, 3:5] = 1
        patches = patches_from_mask(_projected_raster(grid), connectivity=4)
        assert len(patches) == 2
        assert sorted(patches["pixel_count"]) == [4, 4]

    def test_rejects_bad_connectivity(self):
        with pytest.raises(ValueError, match="connectivity"):
            patches_from_mask(_projected_raster(np.ones((3, 3), "uint8")), connectivity=6)

    def test_rejects_three_dimensional_input(self):
        da = _projected_raster(np.ones((3, 3), "uint8")).expand_dims(time=[0])
        with pytest.raises(ValueError, match="2-D"):
            patches_from_mask(da)


class TestArea:
    def test_projected_30m_pixels_are_point_09_hectares(self):
        grid = np.zeros((12, 12), dtype="uint8")
        grid[2:12, 2:12] = 1  # 100 pixels
        patches = patches_from_mask(_projected_raster(grid, res=30.0))
        assert len(patches) == 1
        assert patches["pixel_count"].iloc[0] == 100
        assert patches["area_ha"].iloc[0] == pytest.approx(9.0, rel=1e-9)

    def test_geographic_pixels_near_equator_are_about_5m_square(self):
        grid = np.zeros((12, 12), dtype="uint8")
        grid[2:12, 2:12] = 1
        patches = patches_from_mask(_geographic_raster(grid))
        per_pixel = patches["area_ha"].iloc[0] / patches["pixel_count"].iloc[0]
        # A ~5 m cell is a few thousandths of a hectare; degrees squared would
        # be off by many orders of magnitude.
        assert 0.002 < per_pixel < 0.003

    def test_min_area_ha_drops_the_small_blob_on_a_geographic_grid(self):
        grid = np.zeros((20, 20), dtype="uint8")
        grid[1, 1] = 1  # single pixel, ~0.0025 ha
        grid[5:15, 5:15] = 1  # 100 pixels, ~0.25 ha
        da = _geographic_raster(grid)

        assert len(patches_from_mask(da)) == 2

        kept = patches_from_mask(da, min_area_ha=0.01)
        assert len(kept) == 1
        assert kept["pixel_count"].iloc[0] == 100
        assert list(kept["patch_id"]) == [1]

    def test_min_area_ha_drops_the_small_blob_on_a_projected_grid(self):
        grid = np.zeros((20, 20), dtype="uint8")
        grid[1, 1] = 1  # 0.09 ha
        grid[5:15, 5:15] = 1  # 9.0 ha
        da = _projected_raster(grid, res=30.0)
        kept = patches_from_mask(da, min_area_ha=1.0)
        assert len(kept) == 1
        assert kept["area_ha"].iloc[0] == pytest.approx(9.0)

    def test_rejects_non_metre_projected_crs(self):
        grid = np.ones((4, 4), dtype="uint8")
        # EPSG:2225 is a US survey foot state plane grid.
        da = _projected_raster(grid, res=100.0, x0=2000000.0, y0=500000.0, crs="EPSG:2225")
        with pytest.raises(ValueError, match="axis units"):
            patches_from_mask(da)


class TestMaskHandling:
    def test_fill_value_pixels_are_background(self):
        grid = np.zeros((6, 6), dtype="int16")
        grid[1:3, 1:3] = 1
        grid[4, 4] = 255  # nodata sentinel
        da = _projected_raster(grid)
        assert len(patches_from_mask(da)) == 2
        assert len(patches_from_mask(da, fill_value=255)) == 1

    def test_attrs_become_constant_columns(self):
        grid = np.zeros((5, 5), dtype="uint8")
        grid[1:3, 1:3] = 1
        patches = patches_from_mask(
            _projected_raster(grid), attrs={"layer": "change", "step": 2}
        )
        assert list(patches["layer"]) == ["change"]
        assert list(patches["step"]) == [2]

    def test_crs_is_carried_onto_the_output(self):
        grid = np.zeros((5, 5), dtype="uint8")
        grid[1:3, 1:3] = 1
        patches = patches_from_mask(_geographic_raster(grid))
        assert patches.crs.to_epsg() == 4326

    def test_empty_mask_returns_no_rows(self):
        patches = patches_from_mask(_projected_raster(np.zeros((5, 5), "uint8")))
        assert len(patches) == 0
        assert "area_ha" in patches.columns

    def test_categorical_records_the_matched_values(self):
        grid = np.zeros((6, 6), dtype="int16")
        grid[1:3, 1:3] = 3
        grid[4:6, 0:2] = 7
        da = _projected_raster(grid)

        one = patches_from_categorical(da, 3)
        assert len(one) == 1
        assert list(one["class_values"]) == ["3"]

        both = patches_from_categorical(da, [3, 7])
        assert len(both) == 2
        assert set(both["class_values"]) == {"3,7"}


class TestOverTime:
    def _cube(self, chunked=False):
        grid = np.zeros((3, 10, 10), dtype="int16")
        grid[0, 1:3, 1:3] = 1
        grid[1, 5:7, 1:3] = 1
        grid[1, 5:7, 6:8] = 1
        grid[2, 8:10, 8:10] = 1
        x = -60.0 + (np.arange(10) + 0.5) * DEG_5M
        y = 0.0 - (np.arange(10) + 0.5) * DEG_5M
        da = xr.DataArray(
            grid,
            coords={"time": [2021, 2022, 2023], "y": y, "x": x},
            dims=["time", "y", "x"],
        )
        da = da.rio.write_crs("EPSG:4326")
        return da.chunk({"time": 1}) if chunked else da

    def test_time_column_and_counts(self):
        patches = patches_over_time(self._cube(), 1)
        assert len(patches) == 4
        assert list(patches["time"]) == [2021, 2022, 2022, 2023]
        assert list(patches["patch_id"]) == [1, 2, 3, 4]
        assert patches.crs.to_epsg() == 4326

    def test_rejects_missing_time_dim(self):
        with pytest.raises(ValueError, match="not a dimension"):
            patches_over_time(self._cube(), 1, time_dim="year")

    def test_only_one_step_is_materialized_at_a_time(self, monkeypatch):
        """The per-step helper must only ever see a single 2-D slice.

        Dask does not expose a post-hoc record of peak materialization, so the
        check is on the shape handed to the polygonizer: if the whole cube were
        loaded up front it would arrive 3-D.
        """
        from ctreeskit.xr_analyzer import xr_vectorize_module as mod

        seen = []
        original = mod.patches_from_mask

        def spy(mask, **kwargs):
            seen.append(mask.shape)
            return original(mask, **kwargs)

        monkeypatch.setattr(mod, "patches_from_mask", spy)
        patches = mod.patches_over_time(self._cube(chunked=True), 1)

        assert len(patches) == 4
        assert seen == [(10, 10), (10, 10), (10, 10)]


class TestMergePatches:
    def _two_patches(self, gap_pixels):
        """Two 2x2 blobs on a 30 m projected grid separated by gap_pixels cells."""
        width = 4 + gap_pixels + 2
        grid = np.zeros((6, width), dtype="uint8")
        grid[2:4, 0:2] = 1
        grid[2:4, 2 + gap_pixels: 4 + gap_pixels] = 1
        return _projected_raster(grid, res=10.0)

    def test_merges_at_a_gap_wider_than_the_separation(self):
        patches = patches_from_mask(self._two_patches(gap_pixels=2))  # 20 m apart
        assert len(patches) == 2
        events, membership = merge_patches(patches, gap_m=30.0)
        assert len(events) == 1
        assert events["patch_count"].iloc[0] == 2
        assert len(membership) == 2

    def test_does_not_merge_at_a_gap_narrower_than_the_separation(self):
        patches = patches_from_mask(self._two_patches(gap_pixels=2))  # 20 m apart
        events, membership = merge_patches(patches, gap_m=10.0)
        assert len(events) == 2
        assert sorted(membership["patch_id"]) == [1, 2]

    def test_merges_across_a_20m_gap_on_a_geographic_grid(self):
        # Two blobs three ~5 m cells apart, i.e. ~15 m, on a lat/lon grid.
        grid = np.zeros((6, 10), dtype="uint8")
        grid[2:4, 0:2] = 1
        grid[2:4, 5:7] = 1
        patches = patches_from_mask(_geographic_raster(grid))
        assert len(patches) == 2
        assert len(merge_patches(patches, gap_m=30.0)[0]) == 1
        assert len(merge_patches(patches, gap_m=5.0)[0]) == 2

    def test_group_by_prevents_merging_across_steps(self):
        grid = np.zeros((2, 6, 6), dtype="int16")
        grid[0, 2:4, 0:2] = 1
        grid[1, 2:4, 2:4] = 1  # adjacent in space, different step
        x = 500000.0 + (np.arange(6) + 0.5) * 10.0
        y = 4000000.0 - (np.arange(6) + 0.5) * 10.0
        da = xr.DataArray(
            grid, coords={"time": [2021, 2022], "y": y, "x": x},
            dims=["time", "y", "x"],
        ).rio.write_crs("EPSG:32633")
        patches = patches_over_time(da, 1)
        assert len(patches) == 2

        assert len(merge_patches(patches, gap_m=30.0)[0]) == 1
        grouped, membership = merge_patches(patches, gap_m=30.0, group_by=["time"])
        assert len(grouped) == 2
        assert sorted(grouped["time"]) == [2021, 2022]
        assert len(membership) == 2

    def test_event_geometry_is_the_union_not_the_buffered_hull(self):
        patches = patches_from_mask(self._two_patches(gap_pixels=2))
        events, _ = merge_patches(patches, gap_m=30.0)
        union = patches.geometry.union_all()
        assert events.geometry.iloc[0].equals(union)
        # The buffered hull would be one connected piece covering the gap.
        assert events.geometry.iloc[0].area == pytest.approx(union.area)
        assert events["area_ha"].iloc[0] == pytest.approx(patches["area_ha"].sum())
        assert events["pixel_count"].iloc[0] == patches["pixel_count"].sum()

    def test_membership_covers_every_patch_exactly_once(self):
        grid = np.zeros((20, 20), dtype="uint8")
        grid[1:3, 1:3] = 1
        grid[1:3, 8:10] = 1
        grid[15:17, 15:17] = 1
        patches = patches_from_mask(_projected_raster(grid, res=10.0))
        events, membership = merge_patches(patches, gap_m=20.0)
        assert sorted(membership["patch_id"]) == sorted(patches["patch_id"])
        assert set(membership["event_id"]) == set(events["event_id"])
        assert events["patch_count"].sum() == len(patches)

    def test_rejects_negative_gap(self):
        patches = patches_from_mask(self._two_patches(gap_pixels=2))
        with pytest.raises(ValueError, match="gap_m"):
            merge_patches(patches, gap_m=-1.0)


class TestAssignToPolygons:
    def _straddling_event(self):
        grid = np.zeros((10, 20), dtype="uint8")
        grid[2:8, 4:16] = 1
        patches = patches_from_mask(_projected_raster(grid, res=30.0))
        events, _ = merge_patches(patches, gap_m=10.0)
        return events

    def _split_polygons(self, events):
        minx, miny, maxx, maxy = events.total_bounds
        mid = (minx + maxx) / 2
        pad = 1000.0
        return gpd.GeoDataFrame(
            {"unit_id": ["west", "east"]},
            geometry=[
                box(minx - pad, miny - pad, mid, maxy + pad),
                box(mid, miny - pad, maxx + pad, maxy + pad),
            ],
            crs=events.crs,
        )

    def test_event_straddling_two_polygons_yields_two_rows_summing_to_one(self):
        events = self._straddling_event()
        table = assign_to_polygons(events, self._split_polygons(events), polygon_id="unit_id")
        assert len(table) == 2
        assert set(table["unit_id"]) == {"west", "east"}
        assert table["overlap_frac"].sum() == pytest.approx(1.0, rel=1e-6)
        assert table["overlap_area_ha"].sum() == pytest.approx(
            events["area_ha"].iloc[0], rel=1e-2
        )

    def test_min_overlap_frac_filters(self):
        events = self._straddling_event()
        polygons = self._split_polygons(events)
        # Shrink the west polygon so it only clips a sliver of the event.
        minx, miny, maxx, maxy = events.total_bounds
        polygons.loc[0, "geometry"] = box(minx - 1000, miny - 1000, minx + 30, maxy + 1000)
        polygons.loc[1, "geometry"] = box(minx + 30, miny - 1000, maxx + 1000, maxy + 1000)

        unfiltered = assign_to_polygons(events, polygons, polygon_id="unit_id")
        assert len(unfiltered) == 2

        filtered = assign_to_polygons(
            events, polygons, polygon_id="unit_id", min_overlap_frac=0.5
        )
        assert list(filtered["unit_id"]) == ["east"]

    def test_non_overlapping_polygons_are_omitted(self):
        events = self._straddling_event()
        polygons = self._split_polygons(events)
        far = gpd.GeoDataFrame(
            {"unit_id": ["elsewhere"]},
            geometry=[box(1_000_000, 1_000_000, 1_001_000, 1_001_000)],
            crs=events.crs,
        )
        polygons = gpd.GeoDataFrame(
            pd.concat([polygons, far], ignore_index=True),
            geometry="geometry",
            crs=events.crs,
        )
        table = assign_to_polygons(events, polygons, polygon_id="unit_id")
        assert "elsewhere" not in set(table["unit_id"])

    def test_reprojects_polygons_given_in_another_crs(self):
        events = self._straddling_event()
        polygons = self._split_polygons(events).to_crs("EPSG:4326")
        table = assign_to_polygons(events, polygons, polygon_id="unit_id")
        assert len(table) == 2
        assert table["overlap_frac"].sum() == pytest.approx(1.0, rel=1e-3)

    def test_rejects_missing_polygon_id(self):
        events = self._straddling_event()
        with pytest.raises(ValueError, match="no 'missing' column"):
            assign_to_polygons(events, self._split_polygons(events), polygon_id="missing")


class TestGeoParquetRoundTrip:
    def test_round_trip_preserves_crs_and_columns(self, tmp_path):
        grid = np.zeros((10, 10), dtype="uint8")
        grid[1:3, 1:3] = 1
        grid[6:8, 6:8] = 1
        patches = patches_from_mask(
            _geographic_raster(grid), attrs={"layer": "change"}
        )
        path = tmp_path / "patches.parquet"
        write_geoparquet(patches, path)
        restored = read_geoparquet(path)

        assert restored.crs == patches.crs
        assert restored.crs.to_epsg() == 4326
        assert list(restored.columns) == list(patches.columns)
        assert len(restored) == len(patches)
        assert restored["area_ha"].to_list() == pytest.approx(patches["area_ha"].to_list())
        assert all(a.equals(b) for a, b in zip(restored.geometry, patches.geometry))

    def test_refuses_to_write_without_a_crs(self, tmp_path):
        gdf = gpd.GeoDataFrame({"event_id": [1]}, geometry=[box(0, 0, 1, 1)])
        with pytest.raises(ValueError, match="no CRS"):
            write_geoparquet(gdf, tmp_path / "out.parquet")

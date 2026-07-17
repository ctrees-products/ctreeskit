"""Coverage-fraction weighted zonal stats via rasterix (the `zonal` extra).

Follows the documented workflow in docs/xr_analyzer.md: exactextract-backed
per-pixel coverage fractions feed calculate_categorical_area_stats as the
area grid. Requires the `zonal` extra; skipped where its backend is
unavailable (rasterix's exactextract path needs `sparse`, which has no
Python 3.14 support yet).
"""
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import box

from ctreeskit import calculate_categorical_area_stats

exact = pytest.importorskip("rasterix.rasterize.exact")
gpd = pytest.importorskip("geopandas")


@pytest.fixture
def classes():
    """10x10 raster of 0.01-degree pixels near 10N, -60E: class 1 in the
    north half, class 2 in the south half."""
    y = np.linspace(9.995, 9.905, 10)  # descending latitude (north-up)
    x = np.linspace(-59.995, -59.905, 10)
    data = np.zeros((10, 10))
    data[:5, :] = 1
    data[5:, :] = 2
    da = xr.DataArray(data, dims=["y", "x"], coords={"y": y, "x": x},
                      name="classification")
    return da.rio.write_crs("EPSG:4326")


@pytest.fixture
def aoi():
    """Geometry offset half a pixel in both axes: covers a 5x5 pixel block
    with 0.25 corners, 0.5 edges and a fully covered interior (16 pixel
    equivalents in total: rows 2-4 in class 1, rows 5-6 in class 2)."""
    return gpd.GeoDataFrame(
        geometry=[box(-59.975, 9.935, -59.935, 9.975)], crs="EPSG:4326")


def _dense_cover(cover):
    cover_2d = cover.isel(geometry=0)
    return cover_2d.copy(data=cover_2d.data.todense())


def test_coverage_fractions_sum_to_geometry_footprint(classes, aoi):
    cover = _dense_cover(exact.coverage(classes, aoi, coverage_weight="fraction"))
    # geometry area / pixel area = (0.04 * 0.04) / 0.0001
    assert float(cover.sum()) == pytest.approx(16.0, abs=1e-6)
    assert float(cover.max()) == pytest.approx(1.0)


def test_coverage_weighted_area_stats(classes, aoi):
    cover = _dense_cover(exact.coverage(classes, aoi, coverage_weight="fraction"))
    df = calculate_categorical_area_stats(classes, area_ds=cover * 1.0)
    assert df.iloc[0][1] == pytest.approx(10.0, abs=1e-6)
    assert df.iloc[0][2] == pytest.approx(6.0, abs=1e-6)
    assert df.iloc[0]["total_area"] == pytest.approx(16.0, abs=1e-6)


def test_coverage_is_lazy_for_chunked_input(classes, aoi):
    cover = exact.coverage(
        classes.chunk({"y": 5}), aoi, coverage_weight="fraction")
    assert type(cover.data).__name__ == "Array"  # dask, not computed

import pytest
import xarray as xr
import numpy as np
import rioxarray
from ctreeskit import XrZonalStats


@pytest.fixture
def test_arrays():
    """Create sample arrays for testing."""
    # Create coordinates
    coords = {
        'y': np.array([0, 1, 2, 3]),
        'x': np.array([0, 1, 2, 3])
    }

    # Create sample arrays
    categorical = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4]
    ])

    continuous = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [50, 60, 70, 80],
        [55, 65, 75, 85]
    ])

    percentage = np.array([
        [100, 75, 50, 25],
        [80, 60, 40, 20],
        [70, 50, 30, 10],
        [60, 40, 20, 0]
    ])

    area = np.ones((4, 4)) * 0.5  # 0.5 ha per pixel

    # Create DataArrays with CRS
    categorical_da = xr.DataArray(categorical, coords=coords, dims=['y', 'x'])
    continuous_da = xr.DataArray(continuous, coords=coords, dims=['y', 'x'])
    percentage_da = xr.DataArray(percentage, coords=coords, dims=['y', 'x'])
    area_da = xr.DataArray(area, coords=coords, dims=['y', 'x'],
                           attrs={
        'units': 'ha',
        'unit_name': 'hectares',
        'description': 'Area per pixel in hectares'
    })

    # Add CRS
    for da in [categorical_da, continuous_da, percentage_da, area_da]:
        da.rio.write_crs("EPSG:4326", inplace=True)

    return {
        'categorical_da': categorical_da,
        'continuous_da': continuous_da,
        'percentage_da': percentage_da,
        'area_da': area_da
    }


class TestXrZonalStats:
    """Test class for XrZonalStats."""

    def test_initialization(self, test_arrays):
        """Test proper initialization of XrZonalStats."""
        stats = XrZonalStats(**test_arrays)
        assert stats.categorical_da is not None
        assert stats.continuous_da is not None
        assert stats.percentage_da is not None
        assert stats.area_da is not None

    def test_initialization_with_constant_area(self, test_arrays):
        """Test initialization with constant area value."""
        stats = XrZonalStats(
            categorical_da=test_arrays['categorical_da'],
            continuous_da=test_arrays['continuous_da'],
            area_value=0.5
        )
        assert stats.area_value == 0.5

    def test_crs_mismatch(self, test_arrays):
        """Test CRS mismatch detection."""
        invalid_da = test_arrays['categorical_da'].copy()
        invalid_da.rio.write_crs("EPSG:3857", inplace=True)

        with pytest.raises(ValueError, match="CRS mismatch"):
            XrZonalStats(categorical_da=invalid_da,
                         continuous_da=test_arrays['continuous_da'])

    def test_calculate_categorical_stats(self, test_arrays):
        """Test categorical statistics calculation."""
        stats = XrZonalStats(**test_arrays)
        results = stats.calculate_categorical_stats()

        assert len(results) == 4  # Four unique categories
        for result in results:
            assert "category" in result
            assert "pixel_count" in result
            assert "area_ha" in result

    def test_calculate_continuous_stats(self, test_arrays):
        """Test continuous statistics calculation."""
        stats = XrZonalStats(**test_arrays)
        results = stats.calculate_continuous_stats()

        expected_keys = ["count", "sum", "mean", "std", "min", "max", "median"]
        assert all(key in results for key in expected_keys)
        assert results["count"] > 0

    def test_calculate_agb_stats(self, test_arrays):
        """Test AGB statistics calculation."""
        stats = XrZonalStats(**test_arrays)
        results = stats.calculate_agb_stats()

        expected_keys = [
            "mean_agb_Mg_per_ha",
            "std_agb_Mg_per_ha",
            "mean_ton_CO2_per_ha",
            "std_ton_CO2_per_ha",
            "total_stock_CO2_Mg",
            "area_ha"
        ]
        assert all(key in results for key in expected_keys)

    def test_calculate_stats_by_category(self, test_arrays):
        """Test statistics calculation by category."""
        stats = XrZonalStats(**test_arrays)
        results = stats.calculate_stats_by_category()

        assert len(results) == 4  # Four categories
        for result in results:
            assert "class" in result
            assert "pixel_count" in result
            assert "mean_agb_Mg_per_ha" in result

    def test_calculate_percentage_area_stats(self, test_arrays):
        """Test percentage area statistics calculation."""
        stats = XrZonalStats(**test_arrays)
        results = stats.calculate_percentage_area_stats()

        assert results.get("primary_area") is not None
        assert results.get("secondary_area") is not None
        assert results.get("primary_area") + \
            results.get("secondary_area") == pytest.approx(8.0)  # Total area

    def test_error_handling(self, test_arrays):
        """Test error handling for missing required data."""
        stats = XrZonalStats()  # Initialize with no data

        with pytest.raises(ValueError):
            stats.calculate_categorical_stats()

        with pytest.raises(ValueError):
            stats.calculate_continuous_stats()

        with pytest.raises(ValueError):
            stats.calculate_percentage_area_stats()

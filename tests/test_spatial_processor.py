import pytest
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from ctreeskit import XrSpatialProcessor, Units


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    # Create sample 4x4 array with no band dimension
    data = np.ones((4, 4))

    coords = {
        'y': np.linspace(9.5, 12.5, 4),
        'x': np.linspace(9.5, 12.5, 4)
    }

    da = xr.DataArray(
        data,
        coords=coords,
        dims=['y', 'x']
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    return da


@pytest.fixture
def sample_geometry():
    """Create a sample geometry for testing."""
    polygon = box(10, 10, 12, 12)
    return gpd.GeoDataFrame(
        geometry=[polygon],
        crs="EPSG:4326"
    )


class TestXrSpatialProcessor:
    """Test class for XrSpatialProcessor."""

    def test_init_with_dataset(self, sample_dataset):
        """Test initialization with just dataset."""
        processor = XrSpatialProcessor(sample_dataset)
        assert processor.da.equals(sample_dataset)
        assert processor.geom is None
        assert processor.geom_bbox is None

    def test_init_with_geometry(self, sample_dataset, sample_geometry):
        """Test initialization with dataset and geometry."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        assert len(processor.geom) == 1
        assert processor.geom_bbox is not None
        assert processor.da.equals(sample_dataset)

    def test_init_with_invalid_crs(self, sample_dataset):
        """Test initialization with mismatched CRS."""
        invalid_geom = gpd.GeoDataFrame(
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:3857"
        )
        processor = XrSpatialProcessor(sample_dataset, invalid_geom)
        assert processor.geom is not None
        assert processor.geom[0].bounds == invalid_geom.to_crs(
            "EPSG:4326").geometry.iloc[0].bounds

    def test_init_with_units(self, sample_dataset):
        """Test initialization with different units."""
        # Test string unit specification
        processor_ha = XrSpatialProcessor(sample_dataset, unit="ha")
        assert processor_ha.unit.symbol == "ha"

        # Test Units class constant
        processor_km2 = XrSpatialProcessor(sample_dataset, unit=Units.KM2)
        assert processor_km2.unit.symbol == "km²"

        # Test default unit
        processor_default = XrSpatialProcessor(sample_dataset)
        assert processor_default.unit.symbol == "ha"

    def test_create_binary_geom_mask_da(self, sample_dataset, sample_geometry):
        """Test binary geometry mask creation."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        mask = processor.create_binary_geom_mask_da()
        assert isinstance(mask, xr.DataArray)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask.values)) <= {0, 1}

    def test_create_weighted_geom_mask_da(self, sample_dataset, sample_geometry):
        """Test weighted geometry mask creation."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        weights = processor.create_weighted_geom_mask_da()

        # Test basic properties
        assert isinstance(weights, xr.DataArray)
        assert weights.dims == ('y', 'x')

        # Test value ranges
        weight_values = weights.values
        assert np.all(weight_values >= 0)  # Changed from .all()
        assert np.all(weight_values <= 1)  # Changed from .all()

    def test_create_area_mask_da(self, sample_dataset):
        """Test area calculation."""
        processor = XrSpatialProcessor(sample_dataset)
        areas = processor.create_area_mask_da()
        assert isinstance(areas, xr.DataArray)
        assert (areas.values > 0).all()
        assert areas.attrs['units'] == 'ha'

    def test_create_weighted_area_geom_mask_da(self, sample_dataset, sample_geometry):
        """Test weighted area calculation."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        weighted_areas = processor.create_weighted_area_geom_mask_da()
        assert isinstance(weighted_areas, xr.DataArray)
        assert weighted_areas.attrs['units'] == 'ha'
        assert (weighted_areas.values >= 0).all()

    def test_create_clipped_da_vector(self, sample_dataset, sample_geometry):
        """Test vector clipping."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        clipped = processor.create_clipped_da_vector()
        # Test basic properties
        assert isinstance(clipped, xr.DataArray)
        assert clipped.dims == ('y', 'x')

        # Test that some values are masked (0)
        masked_values = clipped.values
        assert np.all(masked_values >= 0)  # Changed from .all()
        assert np.all(masked_values <= 1)  # Changed from .all()

    def test_create_clipped_da_raster(self, sample_dataset, sample_geometry):
        """Test raster clipping."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)
        clipped = processor.create_clipped_da_raster()
        assert isinstance(clipped, xr.DataArray)
        masked_values = clipped.values
        assert np.all(masked_values >= 0)  # Changed from .all()
        assert np.all(masked_values <= 1)  # Changed from .all()

    def test_compare_clipped(self, sample_dataset, sample_geometry):
        """Test raster and vector clipping results."""
        processor = XrSpatialProcessor(sample_dataset, sample_geometry)

        # Get both clipped versions
        clipped_ra = processor.create_clipped_da_raster()
        clipped_va = processor.create_clipped_da_vector()

        # Print detailed information
        print("\nRaster clipping info:")
        print(f"Size: {clipped_ra.size}")
        print(f"Values: \n{clipped_ra.values}")
        print(f"Non-null count: {np.sum(~np.isnan(clipped_ra.values))}")

        print("\nVector clipping info:")
        print(f"Size: {clipped_va.size}")
        print(f"Values: \n{clipped_va.values}")
        print(f"Non-null count: {np.sum(~np.isnan(clipped_va.values))}")

        # Check binary mask
        binary_mask = processor.create_binary_geom_mask_da()
        print("\nBinary mask info:")
        print(f"Values: \n{binary_mask.values}")
        print(f"Sum of ones: {np.sum(binary_mask.values == 1)}")

        # Modified assertions
        assert clipped_ra.size > 0, "Raster clipped array has size 0"
        assert np.any(~np.isnan(clipped_ra.values)
                      ), "All values are NaN in raster clip"
        assert np.array_equal(
            np.isnan(clipped_ra.values),
            np.isnan(clipped_va.values)
        ), "Mask patterns differ"

    def test_error_no_geometry(self, sample_dataset):
        """Test error handling when no geometry is provided."""
        processor = XrSpatialProcessor(sample_dataset)
        with pytest.raises(ValueError):
            processor.create_binary_geom_mask_da()

from typing import Dict, List, Union, Optional, Tuple
import xarray as xr
import numpy as np
from .common import Units, AreaUnit


class XrZonalStats:
    """
    Calculate zonal statistics for categorical and continuous data with optional area weighting.

    This class handles calculations of statistics within zones defined by categorical data,
    continuous value analysis, and area-weighted calculations. It assumes input data arrays
    are pre-masked to the region of interest.

    Key Features:
        - Calculate statistics by category/zone
        - Compute continuous data statistics (mean, std, etc.)
        - Calculate AGB and carbon statistics with scaling
        - Handle area-weighted calculations using pixel-specific or constant areas
        - Process percentage-based coverage statistics
        - Support multiple area units (m², ha, km²)

    Attributes:
        AGB_SCALING_FACTOR (int): Default scaling factor for AGB calculations (10)
        categorical_da (xr.DataArray): Categorical data defining zones
        continuous_da (xr.DataArray): Continuous values for statistics
        percentage_da (xr.DataArray): Percentage coverage values (0-100)
        area_da (xr.DataArray): Per-pixel areas with units defined in attrs
        area_value (float): Constant area per pixel
        area_unit (AreaUnit): Unit for area_value calculations

    Example Usage:
        ```python
        # Using area_da with its own units
        areas = xr.DataArray(
            data,
            attrs={'units': 'km²'}
        )
        stats = XrZonalStats(
            categorical_da=forest_types,
            continuous_da=biomass_data,
            area_da=areas
        )

        # Using constant area value with specified unit
        stats = XrZonalStats(
            categorical_da=forest_types,
            continuous_da=biomass_data,
            area_value=100,
            area_unit=Units.HA  # or "ha" as string
        )
        ```

    Notes:
        - Area units are preserved from area_da or specified by area_unit
        - All values must be in the same CRS
        - Negative values in continuous data are treated as no-data
        - Category value 0 is treated as no-data in categorical arrays
        - area_unit only applies when using area_value
        - area_da must include 'units' in its attributes
    """

    # Class-level constant for AGB scaling
    AGB_SCALING_FACTOR = 10

    def __init__(
        self,
        categorical_da: Optional[xr.DataArray] = None,
        continuous_da: Optional[xr.DataArray] = None,
        percentage_da: Optional[xr.DataArray] = None,
        area_da: Optional[xr.DataArray] = None,
        area_value: Optional[float] = None,
        area_unit: Union[str, AreaUnit] = "ha"
    ):
        """
        Initialize ZonalStats with input data arrays and area information.

        Args:
            categorical_da: Categorical mask data
            continuous_da: Continuous value data
            percentage_da: Percentage coverage values (0-100)
            area_da: Area array in specifed unit
            area_value: Single area value in specified unit
            area_value: Single area value
            area_unit: Unit for area_value (default: "ha"). Can be string ("m2", "ha", "km2") 
                or Units constant (Units.HA). Only applies to area_value, (inhereted from area_da)
        Raises:
            ValueError: If input DataArrays have different CRS
        """
        # Store all DataArrays in a dict for validation
        data_arrays = {
            'categorical_da': categorical_da,
            'continuous_da': continuous_da,
            'percentage_da': percentage_da,
            'area_da': area_da
        }

        # Get CRS from first non-None DataArray
        reference_crs = None
        for name, da in data_arrays.items():
            if da is not None:
                if not hasattr(da, 'rio'):
                    raise ValueError(
                        f"{name} must be a rioxarray DataArray with CRS information")
                reference_crs = da.rio.crs
                break

        # Verify all DataArrays have the same CRS
        if reference_crs:
            for name, da in data_arrays.items():
                if da is not None:
                    if da.rio.crs != reference_crs:
                        raise ValueError(
                            f"CRS mismatch: {name} has CRS {da.rio.crs}, "
                            f"expected {reference_crs}"
                        )

        # Store attributes after validation
        self.categorical_da = categorical_da
        self.continuous_da = continuous_da
        self.percentage_da = percentage_da
        self.area_da = area_da
        self.area_value = None

        if self.area_da is not None and 'units' in area_da.attrs:
            self.area_unit = Units.get_unit(area_da.attrs['units'])

        if area_value is not None and self.area_da is None:
            # Handle area value and its unit
            self.area_unit = Units.get_unit(
                area_unit) if area_value is not None else None
            self.area_value = area_value

    def _get_categories(self) -> np.ndarray:
        """Get unique categories excluding NaN and 0 values."""
        if self.categorical_da is None:
            raise ValueError("Categorical data array is required")

        # Mask values <= 0 and get values
        masked_data = self.categorical_da.where(self.categorical_da > 0).values

        # Get unique values excluding NaN
        categories = np.unique(masked_data)
        categories = categories[~np.isnan(categories)]

        return categories.astype(int)

    def calculate_categorical_stats(self) -> List[Dict[str, Union[int, float]]]:
        """
        Calculate areas or pixel counts for each category.

        Returns:
            List of dictionaries containing:
                - category: category value
                - pixel_count: number of pixels
                - area_{unit}: area in specified unit
        """
        results = []
        for category in self._get_categories():
            category_mask = self.categorical_da == category

            result_dict = {
                "category": int(category),
                "pixel_count": int(category_mask.sum())
            }

            if self.area_da is not None:
                area_masked = self.area_da.where(self.area_da > 0, 0)
                area_key = f"area_{self.area_da.attrs['units']}"
                result_dict[area_key] = float(
                    (category_mask * area_masked).sum())
            elif self.area_value is not None:
                area_key = f"area_{self.area_unit.symbol}"
                result_dict[area_key] = float(
                    category_mask.sum()) * self.area_value

            results.append(result_dict)

        return results

    def calculate_continuous_stats(self, scaling_factor: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate basic statistics for continuous data.

        Args:
            scaling_factor: Optional scaling factor to apply to values

        Returns:
            Dictionary with basic statistics (count, sum, mean, std, min, max, median)
        """
        if self.continuous_da is None:
            raise ValueError("Continuous data array is required")

        if self.continuous_da.count() == 0:
            return {
                "count": 0,
                "sum": None,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None
            }

        valid_data = self.continuous_da.where(self.continuous_da > 0)
        if scaling_factor:
            valid_data = valid_data / scaling_factor

        return {
            "count": int(valid_data.count()),
            "sum": float(valid_data.sum()),
            "mean": float(valid_data.mean()),
            "std": float(valid_data.std()),
            "min": float(valid_data.min()),
            "max": float(valid_data.max()),
            "median": float(valid_data.median())
        }

    def calculate_agb_stats(self, scaling_factor=AGB_SCALING_FACTOR) -> Dict[str, float]:
        """
        Calculate AGB and carbon statistics.

        Args:
            scaling_factor: Scaling factor for AGB values

        Returns:
            Dictionary with AGB and carbon statistics
        """
        if self.continuous_da is None:
            raise ValueError("Continuous data array is required")

        if self.area_unit != Units.HA:
            raise ValueError(
                f"Area unit must be hectares for AGB calculations, got {self.area_unit.name}")

        if self.continuous_da.count() == 0:
            result = {
                "mean_agb_Mg_per_ha": None,
                "std_agb_Mg_per_ha": None,
                "mean_ton_CO2_per_ha": None,
                "std_ton_CO2_per_ha": None
            }
            if self.area_da is not None or self.area_value is not None:
                result.update({
                    "total_stock_CO2_Mg": None,
                    "upper_total_stock_CO2_Mg": None,
                    "lower_total_stock_CO2_Mg": None,
                    "area_ha": 0 if self.area_value is not None else float(
                        self.area_da.where(self.area_da > 0, 0).sum())
                })
            return result

        # Calculate basic statistics
        mean_agb = float(self.continuous_da.mean() / scaling_factor)
        std_agb = float(self.continuous_da.std() / scaling_factor)

        # Calculate carbon and CO2 values
        mean_carbon = mean_agb * 0.5
        std_carbon = std_agb * 0.5
        mean_ton_CO2_ha = mean_carbon * (44 / 12)
        std_ton_CO2_ha = std_carbon * (44 / 12)

        result = {
            "mean_agb_Mg_per_ha": mean_agb,
            "std_agb_Mg_per_ha": std_agb,
            "mean_ton_CO2_per_ha": mean_ton_CO2_ha,
            "std_ton_CO2_per_ha": std_ton_CO2_ha
        }

        # Calculate area-dependent statistics if area is provided
        total_area = None
        if self.area_da is not None:
            area_masked = self.area_da.where(self.area_da > 0, 0)
            total_area = float(area_masked.sum())
        elif self.area_value is not None:
            total_area = self.area_value

        if total_area is not None:
            result.update({
                "total_stock_CO2_Mg": mean_ton_CO2_ha * total_area,
                "upper_total_stock_CO2_Mg": (mean_ton_CO2_ha + std_ton_CO2_ha) * total_area,
                "lower_total_stock_CO2_Mg": max(
                    (mean_ton_CO2_ha - std_ton_CO2_ha) * total_area, 0),
                "area_ha": total_area
            })

        return result

    def calculate_stats_by_category(
        self,
        scaling_factor: float = None,
        agb: bool = True
    ) -> List[Dict[str, Union[str, int, float]]]:
        """
        Calculate statistics for each category.

        Args:
            scaling_factor: Scaling factor for values
            agb: Whether to calculate AGB stats (True) or basic stats (False)

        Returns:
            List of dictionaries with statistics per category.
            Area values will use units from area_da.attrs['units'] or area_unit.symbol
        """
        if self.categorical_da is None or self.continuous_da is None:
            raise ValueError(
                "Both categorical and continuous data arrays are required")

        results = []
        for category in self._get_categories():
            mask = self.categorical_da == category
            result_dict = {"class": int(category)}
            result_dict["pixel_count"] = int(mask.sum())

            # Calculate area if provided
            area_value = None
            if self.area_da is not None:
                area_masked = self.area_da.where(self.area_da > 0, 0)
                area_value = float((mask * area_masked).sum())
                area_key = f"area_{self.area_da.attrs['units']}"
                result_dict[area_key] = area_value
            elif self.area_value is not None:
                area_value = float(mask.sum()) * self.area_value
                area_key = f"area_{self.area_unit.symbol}"
                result_dict[area_key] = area_value

            # Calculate statistics for masked continuous data
            masked_values = self.continuous_da.where(
                mask & (self.continuous_da > 0))

            # Pass area information to temp_stats
            temp_stats = XrZonalStats(
                continuous_da=masked_values,
                area_value=area_value,
                area_unit=self.area_unit if self.area_value is not None
                else self.area_da.attrs['units'] if self.area_da is not None
                else None
            )

            if agb:
                stats = temp_stats.calculate_agb_stats(self.AGB_SCALING_FACTOR)
            else:
                stats = temp_stats.calculate_continuous_stats(scaling_factor)

            result_dict.update(stats)
            results.append(result_dict)

        return results

    def calculate_percentage_area_stats(self) -> Dict[str, float]:
        """
        Calculate primary and secondary area based on percentage values.

        A percentage value of 100 means the pixel is fully primary, while 0 means fully secondary.
        Each pixel's area contribution is weighted by its percentage value.

        Args:
            self.percentage_da (xr.DataArray): Percentage values (0-100)
            self.area_da (xr.DataArray, optional): Area per pixel with units in attrs
            self.area_value (float, optional): Area value in specified unit
            self.area_unit (AreaUnit, optional): Unit for area_value

        Returns:
            Dict[str, float]: Dictionary containing:
                - primary_area_{unit}: Area weighted by percentage/100
                - secondary_area_{unit}: Area weighted by (1 - percentage/100)
                - unit_name: Name of the area unit used
                - unit_symbol: Symbol of the area unit used

        Example:
            For a pixel with 75% coverage and area of 1 km²:
            {
                'primary_area_km²': 0.75,
                'secondary_area_km²': 0.25,
                'unit_name': 'square kilometers',
                'unit_symbol': 'km²'
            }
        """
        if self.percentage_da is None:
            raise ValueError("Percentage data array is required")

        if self.area_da is None and self.area_value is None:
            return {
                'primary_area': None,
                'secondary_area': None,
                'unit_name': None,
                'unit_symbol': None
            }

        # Mask values <= 0
        percent_mask = self.percentage_da.where(self.percentage_da > 0, 0)

        # Handle area information and units
        if self.area_da is not None:
            area_masked = self.area_da.where(self.area_da > 0, 0)
            unit = Units.get_unit(self.area_da.attrs['units'])
        else:
            area_masked = self.area_value
            unit = self.area_unit

        # Convert percentage to ratio (0-1)
        ratio = percent_mask / 100

        # Calculate primary and secondary area
        primary_area = float((ratio * area_masked).sum())
        secondary_area = float(((1 - ratio) * area_masked).sum())

        return {
            'primary_area': primary_area,
            'secondary_area': secondary_area,
            'unit_name': unit.name,
            'unit_symbol': unit.symbol
        }

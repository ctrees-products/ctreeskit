import xarray as xr
import numpy as np
from typing import Optional, Union, List, Dict
from .common import Units, AreaUnit
import pandas as pd


class ZonalStatsProcessor:
    """
    Zonal stats calculations using a fixed area raster (or constant area value).

    This class is initialized with an area_da (or constant area value), and the 
    categorical and continuous inputs are provided separately when calculating stats.
    """

    def __init__(
        self,
        area_da: Optional[xr.DataArray] = None,
        area_value: Optional[float] = None,
        area_unit: Union[str, AreaUnit] = "ha"
    ):
        self.area_da = area_da
        self.area_value = area_value
        if self.area_da is not None and 'units' in self.area_da.attrs:
            self.area_unit = Units.get_unit(self.area_da.attrs['units'])
        elif area_value is not None:
            self.area_unit = Units.get_unit(area_unit)
        else:
            self.area_unit = None
        self._validate_area()

    def _validate_area(self):
        if self.area_da is not None and not hasattr(self.area_da, "rio"):
            raise ValueError("area_da must be a rioxarray DataArray")

    def calculate_area_by_category(
        self,
        categorical_da: xr.DataArray
    ) -> List[Dict]:
        """
        Calculate area for each category given a categorical raster.

        If a "time" dimension is present, calculates values per time slice.
        """
        if categorical_da is None or self.area_da is None:
            raise ValueError("Both categorical and area data are required.")
        categorical_da = categorical_da.rename("category")
        masked_area = self.area_da.where(categorical_da > 0, 0)
        results = []
        # If time exists, group by time and category.
        if "time" in categorical_da.dims:
            # Group by "time"
            for time_val, cat_da_t in categorical_da.groupby("time"):
                # Select the corresponding area slice if available.
                area_da_t = self.area_da.sel(
                    time=time_val) if "time" in self.area_da.dims else self.area_da
                # For this time slice, mask the area using the category values.
                masked_area_t = area_da_t.where(cat_da_t > 0, 0)
                # Group by category.
                grouped = masked_area_t.groupby(cat_da_t)
                # Calculate area sum per group.
                area_sums = grouped.sum()
                # Iterate over groups
                for cat_val in area_sums["category"].data:
                    # Skip if category not positive.
                    if np.isnan(cat_val) or cat_val <= 0:
                        continue
                    # Extract summed area for this category.
                    area_val = float(area_sums.sel(category=cat_val).data)
                    results.append({
                        "time": time_val.item() if hasattr(time_val, "item") else time_val,
                        "category": int(cat_val),
                        f"area_{self.area_unit.symbol}": area_val
                    })
        else:
            # No time dimension: group directly by 'category'
            grouped = masked_area.groupby(categorical_da)
            area_sums = grouped.sum()
            for cat_val in area_sums[categorical_da.name].data:
                if np.isnan(cat_val) or cat_val <= 0:
                    continue
                area_val = float(area_sums.sel(
                    {categorical_da.name: cat_val}).data)
                results.append({
                    "category": int(cat_val),
                    f"area_{self.area_unit.symbol}": area_val
                })
        return results

    def calculate_category_counts(self, ds: xr.Dataset, reshape=True) -> pd.DataFrame:
        """
        Calculate counts of each category per timestep from a Dataset.

        Args:
            ds: xarray Dataset with time dimension and classification values

        Returns:
            pandas DataFrame with time, category, and count columns
        """
        # Create ones array same shape as classification data
        ones = xr.ones_like(ds.classification)

        # Group by time first, then by classification values
        result = []
        for t in ds.time:
            time_slice = ones.sel(time=t)
            counts = (time_slice
                      .groupby(ds.classification.sel(time=t))
                      .sum()
                      .to_dataframe('count')
                      .reset_index()
                      .assign(time=t.values))
        result.append(counts)

        # Combine all time slices
        df = (pd.concat(result, ignore_index=True)
              .astype({
                  'classification': 'int32',
                  'count': 'int32'
              }))

        d = df[['time', 'classification', 'count']]
        if reshape:
            d = d.pivot(index='time', columns='classification',
                        values='count')
        return d

    def calculate_continuous_stats(
        self,
        continuous_da: xr.DataArray,
        scaling_factor: Optional[float] = None
    ) -> Dict:
        """Calculate overall statistics on a continuous raster."""
        if continuous_da is None:
            raise ValueError("Continuous data is required.")
        valid = continuous_da.where(continuous_da > 0)
        if scaling_factor:
            valid = valid / scaling_factor
        stats = {
            "count": int(valid.count()),
            "sum": float(valid.sum()),
            "mean": float(valid.mean()),
            "std": float(valid.std()),
            "min": float(valid.min()),
            "max": float(valid.max()),
            "median": float(valid.median())
        }
        return stats

    def calculate_zonal_value_stats(
        self,
        categorical_da: xr.DataArray,
        continuous_da: xr.DataArray,
        scaling_factor: Optional[float] = None
    ) -> List[Dict]:
        """
        Calculate basic zonal statistics for each category 
        given a categorical and continuous raster.
        """
        if categorical_da is None or continuous_da is None:
            raise ValueError(
                "Both categorical and continuous data are required.")
        categories = np.unique(categorical_da.where(categorical_da > 0).values)
        categories = categories[~np.isnan(categories)].astype(int)
        results = []
        for cat in categories:
            mask = categorical_da == cat
            masked_cont = continuous_da.where(mask & (continuous_da > 0))
            if scaling_factor:
                masked_cont = masked_cont / scaling_factor
            results.append({
                "category": int(cat),
                "count": int(masked_cont.count()),
                "mean": float(masked_cont.mean()),
                "std": float(masked_cont.std()),
                "sum": float(masked_cont.sum())
            })
        return results

    def calculate_density_stats(
        self,
        categorical_da: xr.DataArray,
        continuous_da: xr.DataArray,
        scaling_factor: Optional[float] = None
    ) -> List[Dict]:
        """
        Calculate density (continuous value per unit area) for each category.
        """
        if categorical_da is None or continuous_da is None or (self.area_da is None and self.area_value is None):
            raise ValueError(
                "Categorical, continuous, and area data are required.")
        categories = np.unique(categorical_da.where(categorical_da > 0).values)
        categories = categories[~np.isnan(categories)].astype(int)
        results = []
        for cat in categories:
            mask = categorical_da == cat
            masked_cont = continuous_da.where(mask & (continuous_da > 0))
            cont_sum = float(masked_cont.sum())
            if self.area_da is not None:
                masked_area = self.area_da.where(mask, 0)
                total_area = float(masked_area.sum())
            else:
                total_area = float(mask.sum()) * self.area_value
            density = cont_sum / total_area if total_area else None
            results.append({
                "category": int(cat),
                "total_value": cont_sum,
                "total_area": total_area,
                "density": density,
                "area_unit": self.area_unit.symbol
            })
        return results

    def calculate_multi_categorical_stats(
        self,
        categorical_das: List[xr.DataArray],
        continuous_da: xr.DataArray,
        scaling_factor: Optional[float] = None,
        use_area: bool = False
    ) -> Dict:
        """
        Calculate zonal statistics for the combination of multiple categorical rasters.
        For example, combining land use and wetlands.
        """
        # Combine by taking the intersection of all categorical masks
        combined_mask = categorical_das[0] > 0
        for da in categorical_das[1:]:
            combined_mask = combined_mask & (da > 0)
        if continuous_da is None:
            raise ValueError("Continuous data is required for zonal stats.")
        masked_cont = continuous_da.where(combined_mask & (continuous_da > 0))
        stats = {
            "count": int(masked_cont.count()),
            "mean": float(masked_cont.mean()),
            "std": float(masked_cont.std()),
            "sum": float(masked_cont.sum())
        }
        if use_area:
            if self.area_da is not None:
                masked_area = self.area_da.where(combined_mask, 0)
                total_area = float(masked_area.sum())
            else:
                total_area = float(combined_mask.sum()) * self.area_value
            stats["total_area"] = total_area
            stats["density"] = stats["sum"] / \
                total_area if total_area else None
            stats["area_unit"] = self.area_unit.symbol
        return stats

import time
import gc
from typing import Union, Optional
import xarray as xr
import numpy as np
import dask.array as da
from pyproj import Proj, Geod
import dask
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# _measure function remains the same
def _measure(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    geod = Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance


def create_area_ds_from_degrees_ds_dask(
    input_ds: Union[xr.DataArray, xr.Dataset],
    high_accuracy: Optional[bool] = None,
    output_in_ha: bool = True,
) -> xr.DataArray:
    """
    Create area dataset from degrees dataset with detailed logging.
    """
    start_time = time.time()
    logger.info("Starting area dataset creation")

    try:
        # Get coordinates and ensure they're dask arrays
        lat_center = da.from_array(input_ds.y.values, chunks=2000)
        lon_center = da.from_array(input_ds.x.values, chunks=2000)
        logger.info(
            f"Input dimensions - Latitude: {len(lat_center)}, Longitude: {len(lon_center)}"
        )

        if high_accuracy is None:
            high_accuracy = True
            if -70 <= lat_center[0] <= 70:
                high_accuracy = False
        logger.info(f"Using high accuracy mode: {high_accuracy}")

        # Compute differences with optimized chunking
        diff_x = da.diff(lon_center)
        diff_y = da.diff(lat_center)

        # Create bounds arrays using dask operations
        x_bounds = da.concatenate(
            [
                [lon_center[0] - diff_x[0] / 2],
                lon_center[:-1] + diff_x / 2,
                [lon_center[-1] + diff_x[-1] / 2],
            ]
        ).rechunk(2000)
        y_bounds = da.concatenate(
            [
                [lat_center[0] - diff_y[0] / 2],
                lat_center[:-1] + diff_y / 2,
                [lat_center[-1] + diff_y[-1] / 2],
            ]
        ).rechunk(2000)

        gc.collect()

        if high_accuracy:
            n_y = len(lat_center)
            n_x = len(lon_center)
            cell_heights = np.array(
                [
                    _measure(y_bounds[i], lon_center[0], y_bounds[i + 1], lon_center[0])
                    for i in range(n_y)
                ]
            )
            y_centers = (y_bounds[:-1] + y_bounds[1:]) / 2
            cell_widths = np.array(
                [
                    [_measure(y, x_bounds[j], y, x_bounds[j + 1]) for j in range(n_x)]
                    for y in y_centers
                ]
            )
            grid_area_m2 = cell_heights[:, None] * cell_widths

        else:
            logger.info("Processing with EPSG:6933 approximation method")
            logger.info("Creating meshgrid")

            # Calculate chunk sizes based on available memory
            chunk_size = 2000

            # Process in chunks to avoid memory issues
            def process_chunk(x_chunk, y_chunk):
                p = Proj("EPSG:6933", preserve_units=False)
                x2, y2 = p(longitude=x_chunk, latitude=y_chunk)
                return (x2, y2)  # Return as a tuple

            # Create dask arrays with appropriate chunking
            xv, yv = da.meshgrid(x_bounds, y_bounds, indexing="xy")
            xv = xv.rechunk((chunk_size, chunk_size))
            yv = yv.rechunk((chunk_size, chunk_size))

            # Process chunks in parallel
            x2_chunks = []
            y2_chunks = []

            # Calculate number of chunks in each dimension
            n_chunks_y = (xv.shape[0] + chunk_size - 1) // chunk_size
            n_chunks_x = (xv.shape[1] + chunk_size - 1) // chunk_size

            for i in range(n_chunks_y):
                row_x_chunks = []
                row_y_chunks = []
                for j in range(n_chunks_x):
                    # Calculate chunk boundaries
                    y_start = i * chunk_size
                    y_end = min((i + 1) * chunk_size, xv.shape[0])
                    x_start = j * chunk_size
                    x_end = min((j + 1) * chunk_size, xv.shape[1])

                    # Get chunks
                    x_chunk = xv[y_start:y_end, x_start:x_end]
                    y_chunk = yv[y_start:y_end, x_start:x_end]

                    # Create delayed task for each chunk
                    delayed_result = dask.delayed(process_chunk)(x_chunk, y_chunk)

                    # Create separate delayed objects for x and y
                    x2_delayed = dask.delayed(lambda x: x[0])(delayed_result)
                    y2_delayed = dask.delayed(lambda x: x[1])(delayed_result)

                    # Convert to dask arrays with correct shape
                    chunk_shape = (y_end - y_start, x_end - x_start)
                    row_x_chunks.append(
                        da.from_delayed(x2_delayed, shape=chunk_shape, dtype=float)
                    )
                    row_y_chunks.append(
                        da.from_delayed(y2_delayed, shape=chunk_shape, dtype=float)
                    )

                # Concatenate chunks in x direction
                if row_x_chunks:
                    x2_chunks.append(da.concatenate(row_x_chunks, axis=1))
                    y2_chunks.append(da.concatenate(row_y_chunks, axis=1))

            # Concatenate chunks in y direction
            if x2_chunks:
                x2 = da.concatenate(x2_chunks, axis=0)
                y2 = da.concatenate(y2_chunks, axis=0)
            else:
                raise ValueError("No chunks were processed successfully")

            logger.info("Computing grid differences")
            dx = da.subtract(x2[:-1, 1:], x2[:-1, :-1])
            dy = da.subtract(y2[1:, :-1], y2[:-1, :-1])

            grid_area_m2 = da.abs(dx * dy)
            grid_area_m2 = grid_area_m2.rechunk("auto")
            logger.info("Grid area computation complete")

        # Final conversions and DataArray creation
        logger.info("Preparing final output")
        conversion = 1e-4 if output_in_ha else 1.0
        converted_grid_area = da.multiply(grid_area_m2, conversion)
        converted_grid_area = converted_grid_area.rechunk("auto")
        unit = "ha" if output_in_ha else "m²"
        method = "geodesic distances" if high_accuracy else "EPSG:6933 approximation"

        result = xr.DataArray(
            converted_grid_area,
            coords={"y": input_ds.y, "x": input_ds.x},
            dims=["y", "x"],
            attrs={
                "units": unit,
                "description": f"Grid cell area in {unit} computed using {method}",
            },
        )

        execution_time = time.time() - start_time
        logger.info(
            f"Area dataset creation completed successfully in {execution_time:.2f} seconds"
        )
        logger.info(f"Final array shape: {result.shape}")
        gc.collect()
        return result

    except Exception as e:
        logger.error(
            f"Error in create_area_ds_from_degrees_ds: {str(e)}", exc_info=True
        )
        raise


__all__ = [
    "create_area_ds_from_degrees_ds_dask",
]

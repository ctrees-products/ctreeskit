"""
ingest.py

Annual GeoTIFF/VRT -> Icechunk ingestion connector.

Reads finished per-year raster mosaics (GeoTIFF or VRT) and writes them into an
Arraylake/Icechunk repo with an ``(time, y, x)`` layout:

- Preserve the source's native CRS, dtype and nodata value -- do not upcast or relabel.
- Standardize on rioxarray's CF grid-mapping convention (a ``spatial_ref`` coordinate
  carrying ``crs_wkt``/``GeoTransform``), NOT bespoke GeoZarr ``proj:``/``spatial:`` attrs.
- Rely on Zarr's default behaviour of not writing all-fill (all-nodata) chunks for
  sparsity, instead of a hand-rolled sparsity/resumability ledger.

The connector separates two phases:

1. ``initialize_schema`` allocates the full ``(time, y, x)`` domain once (an empty,
   lazy template written with ``compute=False``), so no data chunks are materialized.
2. ``ingest_year`` fills one year at a time. Years that fall inside the pre-allocated
   time axis are written as region writes into their slot; a year beyond the current
   axis is appended with ``append_dim="time"`` -- which is how future years (2026+) grow
   the dataset.

Every write is placed by coordinate: the source raster's x/y coordinates are located
on the stored grid (same resolution, same origin), so a source that covers only part of
the domain fills just its window, and a source that is off-lattice or extends beyond the
domain fails loudly. To enlarge the domain, ``grow_extent`` resizes the array and
relocates existing chunks with Icechunk's metadata-only ``shift_array`` -- pixel data is
never rewritten, and the previous snapshot stays addressable.

The GeoTIFF/VRT read + Icechunk write mechanics live here (in ``ctreeskit``) so thin
service wrappers only need to load a config and call these methods.
"""

# Standard library imports
import json
import logging
from importlib import resources
from typing import Any, Dict, Optional, Sequence, Tuple, cast

# Third-party library imports
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import zarr
import rioxarray  # noqa: F401  (registers the .rio accessor)
from icechunk.xarray import to_icechunk
from arraylake import Client as arraylakeClient

# Local application/library specific imports
from .common import ArraylakeDatasetConfig

logger = logging.getLogger(__name__)

# Map the config's coarse ``unit_type`` to a concrete numpy dtype when no explicit
# ``dtype`` is given. An explicit ``dtype`` in the variable config always wins so that
# categorical products can preserve their native (often uint8) storage.
_UNIT_TYPE_DTYPE = {"int": "int16", "float": "float32"}


def load_config(name: str) -> Dict[str, Any]:
    """
    Load an *example* dataset-config schema bundled in
    ``ctreeskit/arraylake_tools/datasets_config``.

    Only placeholder/template schemas ship in this public package (e.g.
    ``"categorical_raster_with_x_y_time"``). Real dataset configs reference
    private storage paths and are kept out of this public repo -- keep yours
    wherever suits your setup (an S3 config registry, a local directory). Load
    S3-hosted configs with
    :class:`~ctreeskit.arraylake_tools.common.ArraylakeDatasetConfig`, or pass the
    parsed dict straight to :class:`AnnualRasterIngester`.

    Parameters
    ----------
    name : str
        Config file stem, e.g. ``"categorical_raster_with_x_y_time"`` (with or
        without ``.json``).

    Returns
    -------
    Dict[str, Any]
        The parsed configuration dictionary.
    """
    if not name.endswith(".json"):
        name = f"{name}.json"
    with resources.files(__package__).joinpath("datasets_config", name).open("r") as f:
        return json.load(f)


class AnnualRasterIngester:
    """
    Ingest annual raster mosaics (GeoTIFF/VRT) from S3 into an Arraylake/Icechunk repo.

    The ingester is driven by a dataset configuration dictionary in the same shape used
    by :class:`~ctreeskit.arraylake_tools.common.ArraylakeDatasetConfig`: a single
    ``groups`` entry describing the time axis and one classification/measurement variable
    with an ``s3_path_prefix``/``s3_path_suffix`` from which per-year URIs are built.

    Parameters
    ----------
    config : Dict[str, Any]
        Dataset configuration dictionary (e.g. a parsed
        ``annual_forest_cover_30m.json`` from your private config store).
    token : Optional[str]
        Arraylake API token. If omitted (and no ``client`` is passed), the client falls
        back to the cached credentials from ``arraylake auth login`` / the
        ``ARRAYLAKE_TOKEN`` environment variable.
    bucket_nickname : str
        Bucket-config nickname used when creating the repo (default ``arraylake-datasets``).
    client : Optional[arraylake.Client]
        An already-constructed Arraylake client to reuse (e.g. the one a service-function
        runner built from ``ARRAYLAKE_TOKEN``). Takes precedence over ``token``.
    branch_name : str
        Icechunk branch that ``initialize_schema``/``ingest_year`` read and write
        (default ``"main"``). Point this at a disposable test branch to exercise the
        connector without touching the production branch.
    group : Optional[str]
        Which entry of the config's ``groups`` to operate on. Required when the config
        has more than one group; defaults to the only group otherwise.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        token: Optional[str] = None,
        bucket_nickname: str = "arraylake-datasets",
        client: Optional[Any] = None,
        branch_name: str = "main",
        group: Optional[str] = None,
    ):
        self._configure(config, bucket_nickname, group=group)
        self.branch_name = branch_name
        # Arraylake connectivity: reuse an injected client, else token, else cached creds.
        self.token = token
        if client is not None:
            self.client = client
        else:
            self.client = arraylakeClient(token=token) if token else arraylakeClient()

    def _configure(
        self,
        config: Dict[str, Any],
        bucket_nickname: str = "arraylake-datasets",
        group: Optional[str] = None,
    ) -> None:
        """Parse the dataset config into the attributes the connector operates on."""
        self.config = config
        self.bucket_nickname = bucket_nickname

        self.dataset_name = config.get("dataset_name")
        self.organization = config.get("organization")
        if not self.organization and "repo" not in config:
            raise ValueError(
                "dataset config must include 'organization' (or a full 'repo' name)")
        self.repo_name = config.get(
            "repo", f"{self.organization}/{self.dataset_name}")
        self.crs = config.get("crs", "EPSG:4326")

        # Resolve the group + variable this connector operates on.
        self.group_name, group_cfg = self._resolve_group(config, group)
        self.time_config = group_cfg.get("time")
        # Optional intended domain, as edge coordinates (minx, miny, maxx, maxy) in
        # the dataset CRS. When set, initialize_schema builds the grid from it instead
        # of the template raster's footprint, so the domain can be sized to intent.
        self.extent = group_cfg.get("extent")
        self.variable, self.var_config = self._resolve_variable(group_cfg)

        # Storage characteristics: preserve native dtype/nodata rather than upcasting.
        self.dtype = np.dtype(
            self.var_config.get("dtype")
            or _UNIT_TYPE_DTYPE.get(self.var_config.get("unit_type", "int"), "int16")
        )
        self.nodata = self.var_config.get("nodata", -1)
        self.chunks = config.get("chunks", {"time": 1, "y": 2000, "x": 2000})

        # Source URI construction.
        self.s3_path_prefix = self.var_config["s3_path_prefix"]
        self.s3_path_suffix = self.var_config["s3_path_suffix"]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _resolve_group(
        config: Dict[str, Any], group: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Return the (name, config) of the group this connector ingests."""
        groups = config.get("groups", {})
        if not groups:
            raise ValueError("config has no 'groups' entry to ingest")
        if group is not None:
            if group not in groups:
                raise ValueError(
                    f"group {group!r} not in config; available: {sorted(groups)}")
            return group, groups[group]
        if len(groups) > 1:
            raise ValueError(
                "config has several groups "
                f"{sorted(groups)}; pass group=<name> to choose one.")
        name = next(iter(groups))
        return name, groups[name]

    @staticmethod
    def _resolve_variable(group_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return the (name, config) of the single data variable in a group."""
        variables = {k: v for k, v in group_cfg.items()
                     if k not in ("time", "extent") and isinstance(v, dict)}
        if len(variables) != 1:
            raise ValueError(
                "AnnualRasterIngester expects exactly one variable per group; found "
                f"{sorted(variables)}.")
        name = next(iter(variables))
        return name, variables[name]

    def _year_uri(self, year: int) -> str:
        """Build the S3 URI of the annual mosaic for ``year``."""
        return f"{self.s3_path_prefix}{year}{self.s3_path_suffix}"

    def _years(self) -> list:
        """List of years spanned by the configured time axis."""
        if not self.time_config:
            return []
        return pd.date_range(
            start=self.time_config["start"],
            end=self.time_config["end"],
            freq=self.time_config.get("freq", "YS"),
        ).year.tolist()

    def _repo(self):
        return self.client.get_repo(self.repo_name)

    # ------------------------------------------------------------------ creation

    def create_repo(self) -> None:
        """Get or create the Icechunk repo on Arraylake."""
        self.client.get_or_create_repo(
            name=self.repo_name,
            bucket_config_nickname=self.bucket_nickname,
        )
        logger.info("repo ready: %s (bucket=%s)", self.repo_name, self.bucket_nickname)

    # ------------------------------------------------------------------ schema

    def initialize_schema(self, template_year: Optional[int] = None, overwrite: bool = False) -> str:
        """
        Allocate the full ``(time, y, x)`` domain as an empty, lazy template.

        A template year's mosaic is opened only to read its grid (x/y coordinates, CRS);
        no pixel data is read or written. The resulting empty array is written with
        ``compute=False`` so only coordinates + metadata land in the repo -- data chunks
        are materialized later, one year at a time, by :meth:`ingest_year`.

        Parameters
        ----------
        template_year : Optional[int]
            Year whose mosaic defines the spatial grid. Defaults to the first configured year.
        overwrite : bool
            If True, use ``mode="w"`` (replace an existing array); otherwise ``mode="w-"``
            (fail if it already exists).

        Returns
        -------
        str
            The snapshot id of the initialization commit.
        """
        years = self._years()
        time_cfg = self.time_config
        if not years or time_cfg is None:
            raise ValueError("time config is required to initialize an annual schema")
        template_year = template_year or years[0]

        # Read only the grid from the template mosaic (coordinates are derived from the
        # geotransform, so this does not read pixel data). A single-band GeoTIFF opens
        # as a DataArray.
        template = cast(xr.DataArray, rioxarray.open_rasterio(
            self._year_uri(template_year), masked=False))
        template = template.squeeze("band", drop=True)
        if self.extent is None:
            x = template.x.values
            y = template.y.values
        else:
            # Grid from the configured extent at the template's resolution; the
            # template must sit on that lattice or the config is wrong.
            resx, resy = self._resolution(template.x.values, template.y.values)
            x, y = self._grid_from_extent(self.extent, resx, resy)
            self._index_slice(x, template.x.values)
            self._index_slice(y, template.y.values)
        template.close()

        time = pd.date_range(
            start=time_cfg["start"],
            end=time_cfg["end"],
            freq=time_cfg.get("freq", "YS"),
        )

        shape = (len(time), len(y), len(x))
        # One lazy dask chunk for the whole domain -- never computed; on-disk chunk sizes
        # are set via ``encoding`` below (the xarray region-write template pattern).
        placeholder = da.full(shape, self.nodata, dtype=self.dtype, chunks=shape)
        ds = xr.Dataset(
            {self.variable: (("time", "y", "x"), placeholder)},
            coords={"time": time, "y": y, "x": x},
        )

        # rioxarray CF grid-mapping convention: write the CRS as a spatial_ref coordinate.
        ds = ds.rio.write_crs(self.crs)
        ds = ArraylakeDatasetConfig().add_cf_metadata(ds, self.config, crs=self.crs)

        encoding = {
            self.variable: {
                "chunks": (
                    self.chunks.get("time", 1),
                    self.chunks.get("y", 2000),
                    self.chunks.get("x", 2000),
                ),
                "fill_value": self.nodata,
                "dtype": str(self.dtype),
            }
        }

        session = self._repo().writable_session(self.branch_name)
        ds.drop_encoding().to_zarr(
            session.store,
            group=self.group_name,
            mode="w" if overwrite else "w-",
            encoding=encoding,
            compute=False,
        )
        snapshot = session.commit(
            f"initialize schema {self.group_name}/{self.variable} "
            f"[{years[0]}-{years[-1]}] {self.dtype} nodata={self.nodata}"
        )
        logger.info("initialized schema (snapshot %s) shape=%s dtype=%s",
                    snapshot, shape, self.dtype)
        return snapshot

    # ------------------------------------------------------------------ population

    def ingest_year(
        self,
        year: int,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        tag: Optional[str] = None,
    ) -> str:
        """
        Ingest one year's mosaic into the repo.

        The year is written into its pre-allocated slot on the time axis (region write)
        when it already exists; a year beyond the current axis is appended
        (``append_dim="time"``). Only non-nodata chunks are written to storage.

        The source is placed on the stored grid by coordinate: a mosaic covering part of
        the domain fills only its window; one that is off-lattice, at another resolution,
        or larger than the domain raises ``ValueError`` (grow the domain first with
        :meth:`grow_extent`).

        Parameters
        ----------
        year : int
            Year to ingest; its mosaic URI is built from the configured prefix/suffix.
        bbox : Optional[Tuple[float, float, float, float]]
            Optional ``(minx, miny, maxx, maxy)`` window (in the dataset CRS) to ingest a
            spatial subset -- useful for fast verification. If omitted, the full extent
            is ingested.
        tag : Optional[str]
            If given, create an Icechunk tag pointing at this commit (for a per-publish
            snapshot, per the repo's tag-per-version convention).

        Returns
        -------
        str
            The snapshot id of the ingest commit.
        """
        repo = self._repo()

        # Open the year's mosaic (subset first when a bbox is given, then chunk).
        ychunk = self.chunks.get("y", 2000)
        xchunk = self.chunks.get("x", 2000)
        # A single-band GeoTIFF opens as a DataArray.
        if bbox is None:
            da_year = cast(xr.DataArray, rioxarray.open_rasterio(
                self._year_uri(year), masked=False,
                chunks={"band": 1, "y": ychunk, "x": xchunk}, lock=False,
            )).squeeze("band", drop=True)
        else:
            da_year = cast(xr.DataArray, rioxarray.open_rasterio(
                self._year_uri(year), masked=False,
            )).squeeze("band", drop=True)
            minx, miny, maxx, maxy = bbox
            # y is north-up (descending), so slice high -> low.
            da_year = da_year.sel(x=slice(minx, maxx), y=slice(maxy, miny))
            da_year = da_year.chunk({"y": ychunk, "x": xchunk})

        da_year = da_year.astype(self.dtype)
        ds_year = (
            da_year.to_dataset(name=self.variable)
            .expand_dims(time=[pd.Timestamp(f"{year}-01-01")])
        )
        # The CRS lives once on the array (written at init); don't re-write it per year.
        if "spatial_ref" in ds_year.coords:
            ds_year = ds_year.drop_vars("spatial_ref")
        ds_year = ds_year.drop_encoding()

        # Decide region-write vs append based on the stored time axis.
        stored = xr.open_zarr(
            repo.readonly_session(self.branch_name).store, group=self.group_name,
            consolidated=False, chunks=None,
        )
        stored_years = pd.to_datetime(stored.time.values).year.tolist()
        target = pd.Timestamp(f"{year}-01-01")

        session = repo.writable_session(self.branch_name)
        action = None
        # Locate the source on the stored grid (raises if off-lattice or too large).
        y_slice = self._index_slice(stored.y.values, ds_year.y.values)
        x_slice = self._index_slice(stored.x.values, ds_year.x.values)
        full_extent = (
            y_slice == slice(0, stored.sizes["y"])
            and x_slice == slice(0, stored.sizes["x"])
        )
        if year in stored_years:
            t_idx = stored_years.index(year)
            region = {"time": slice(t_idx, t_idx + 1), "y": y_slice, "x": x_slice}
            # Full-extent writes cover whole storage chunks, so keep the safe-chunk
            # guard (needed for distributed writes). A window may end mid-chunk;
            # align_chunks rechunks the source onto the stored chunk boundaries and
            # the guard is relaxed for that single-writer path.
            to_icechunk(
                ds_year, session, group=self.group_name, region=region,
                align_chunks=True, safe_chunks=full_extent,
            )
            action = f"region {region}"
        elif target > stored.time.values.max():
            if not full_extent:
                raise ValueError(
                    "appending a new year must cover the full extent "
                    f"(source window y={y_slice}, x={x_slice} of "
                    f"{stored.sizes['y']}x{stored.sizes['x']}); drop bbox for "
                    f"year {year} (beyond the initialized axis)")
            to_icechunk(ds_year, session, group=self.group_name, append_dim="time")
            action = "append time"
        else:
            raise ValueError(
                f"year {year} is before the initialized axis and not a slot; "
                "re-initialize the schema to include it")

        snapshot = session.commit(f"ingest {self.variable} {year} ({action})")
        logger.info("ingested %s: %s -> snapshot %s", year, action, snapshot)
        if tag:
            repo.create_tag(tag, snapshot)
            logger.info("tagged snapshot %s as '%s'", snapshot, tag)
        return snapshot

    @staticmethod
    def _index_slice(stored_coord: np.ndarray, subset_coord: np.ndarray) -> slice:
        """Integer slice locating ``subset_coord`` within ``stored_coord``.

        Raises ValueError if the subset is not aligned with the stored grid,
        so an off-grid bbox fails loudly instead of writing pixels shifted by
        up to half a cell.
        """
        start = int(np.abs(stored_coord - subset_coord[0]).argmin())
        stop = start + len(subset_coord)
        window = stored_coord[start:stop]
        cell = (float(np.abs(stored_coord[1] - stored_coord[0]))
                if stored_coord.size > 1 else float("inf"))
        tol = cell * 0.01
        if window.size != subset_coord.size or not np.allclose(
                window, subset_coord, rtol=0, atol=tol):
            raise ValueError(
                "source does not align with the stored grid "
                f"(first coord {subset_coord[0]!r} vs nearest stored "
                f"{stored_coord[start]!r}, tolerance {tol!r}; {subset_coord.size} "
                f"cells requested, {window.size} available). The source must share "
                "the stored resolution and origin and lie inside the stored extent; "
                "use grow_extent() to enlarge the domain.")
        return slice(start, stop)

    # ------------------------------------------------------------------ grid helpers

    @staticmethod
    def _resolution(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Positive (resx, resy) cell sizes of a north-up grid (y descending)."""
        if x.size < 2 or y.size < 2:
            raise ValueError("grid needs at least two cells per axis")
        resx = float(x[1] - x[0])
        resy = float(y[0] - y[1])
        if resx <= 0 or resy <= 0:
            raise ValueError(
                "expected an ascending x and descending (north-up) y axis; got "
                f"resx={resx!r} resy={resy!r}")
        return resx, resy

    @staticmethod
    def _grid_from_extent(
        extent: Sequence[float], resx: float, resy: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Cell-centre x (ascending) and y (descending) arrays for edge ``extent``."""
        minx, miny, maxx, maxy = (float(v) for v in extent)
        nx = AnnualRasterIngester._steps(maxx - minx, resx, "extent width")
        ny = AnnualRasterIngester._steps(maxy - miny, resy, "extent height")
        x = minx + resx * (np.arange(nx) + 0.5)
        y = maxy - resy * (np.arange(ny) + 0.5)
        return x, y

    @staticmethod
    def _steps(delta: float, res: float, what: str) -> int:
        """Number of whole cells in ``delta``; raises if not a whole number."""
        n = delta / res
        if abs(n - round(n)) > 1e-6 * max(1.0, abs(n)):
            raise ValueError(
                f"{what} {delta!r} is not a whole number of cells at resolution "
                f"{res!r} ({n!r} cells); snap it to the lattice.")
        if round(n) < 0:
            raise ValueError(f"{what} is negative ({delta!r})")
        return int(round(n))

    @staticmethod
    def _edges(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) outer edges of a cell-centre grid."""
        resx, resy = AnnualRasterIngester._resolution(x, y)
        return (float(x[0]) - resx / 2, float(y[-1]) - resy / 2,
                float(x[-1]) + resx / 2, float(y[0]) + resy / 2)

    # ------------------------------------------------------------------ extent growth

    def grow_extent(
        self,
        extent: Sequence[float],
        snap_to_chunks: bool = True,
        verify_samples: int = 50,
        seed: int = 0,
    ) -> str:
        """
        Enlarge the stored ``(time, y, x)`` domain to ``extent`` without rewriting data.

        The array is resized and, when the domain grows to the north or west (low index
        side), existing chunk references are relocated with Icechunk's metadata-only
        ``shift_array``. The ``x``/``y`` coordinate arrays (and a ``GeoTransform`` on
        ``spatial_ref``, if present) are rewritten for the new grid. Pixel data is never
        read or rewritten; the pre-growth snapshot remains addressable in the history.

        Parameters
        ----------
        extent : Sequence[float]
            Target outer edges ``(minx, miny, maxx, maxy)`` in the dataset CRS. Must
            contain the current extent and lie on the stored lattice.
        snap_to_chunks : bool
            Chunks can only be shifted by whole chunks, so growth to the north/west must
            be a multiple of the chunk size. If True, such growth is rounded *outward*
            to the next chunk boundary (logged); if False, a misaligned request raises.
        verify_samples : int
            After committing, compare this many randomly sampled pixels of the first time
            step, by coordinate, between the pre- and post-growth snapshots. ``0`` skips.
        seed : int
            Seed for the verification sample.

        Returns
        -------
        str
            The snapshot id of the growth commit.
        """
        repo = self._repo()
        before = repo.lookup_branch(self.branch_name)
        stored = xr.open_zarr(
            repo.readonly_session(self.branch_name).store, group=self.group_name,
            consolidated=False, chunks=None,
        )
        x_old, y_old = stored.x.values, stored.y.values
        resx, resy = self._resolution(x_old, y_old)
        left, bottom, right, top = self._edges(x_old, y_old)
        minx, miny, maxx, maxy = (float(v) for v in extent)
        if minx > left or miny > bottom or maxx < right or maxy < top:
            raise ValueError(
                f"new extent {tuple(extent)} does not contain the stored extent "
                f"{(left, bottom, right, top)}; grow_extent only enlarges the domain.")

        n_west = self._steps(left - minx, resx, "westward growth")
        n_east = self._steps(maxx - right, resx, "eastward growth")
        n_north = self._steps(maxy - top, resy, "northward growth")
        n_south = self._steps(bottom - miny, resy, "southward growth")

        session = repo.writable_session(self.branch_name)
        root = zarr.open_group(session.store, mode="r+")
        group = cast(zarr.Group, root[self.group_name])
        arr = cast(zarr.Array, group[self.variable])
        _, ychunk, xchunk = arr.chunks

        # Low-index growth is realised by shifting chunks, so it must be whole chunks.
        for name, n, chunk in (("northward", n_north, ychunk), ("westward", n_west, xchunk)):
            if n % chunk:
                snapped = int(np.ceil(n / chunk)) * chunk
                if not snap_to_chunks:
                    raise ValueError(
                        f"{name} growth of {n} cells is not a whole number of "
                        f"{chunk}-cell chunks; use {snapped} cells or "
                        "snap_to_chunks=True.")
                logger.info("snapping %s growth %d -> %d cells (chunk %d)",
                            name, n, snapped, chunk)
                if name == "northward":
                    n_north = snapped
                else:
                    n_west = snapped
        if not any((n_west, n_east, n_north, n_south)):
            raise ValueError("stored extent already covers the requested extent")

        new_left = left - n_west * resx
        new_top = top + n_north * resy
        nx = x_old.size + n_west + n_east
        ny = y_old.size + n_north + n_south
        x_new = new_left + resx * (np.arange(nx) + 0.5)
        y_new = new_top - resy * (np.arange(ny) + 0.5)
        new_edges = (new_left, float(y_new[-1]) - resy / 2,
                     float(x_new[-1]) + resx / 2, new_top)

        nt = arr.shape[0]
        arr.resize((nt, ny, nx))
        if n_north or n_west:
            session.shift_array(
                f"/{self.group_name}/{self.variable}",
                (0, n_north // ychunk, n_west // xchunk),
            )
        for name, values in (("y", y_new), ("x", x_new)):
            coord = cast(zarr.Array, group[name])
            coord.resize(values.shape)
            coord[:] = values
        if "spatial_ref" in group:
            sr = cast(zarr.Array, group["spatial_ref"])
            if "GeoTransform" in sr.attrs:
                sr.update_attributes({
                    "GeoTransform": f"{new_left} {resx} 0.0 {new_top} 0.0 {-resy}"})

        snapshot = session.commit(
            f"grow extent {self.group_name}/{self.variable} "
            f"{(left, bottom, right, top)} -> {new_edges} "
            f"(+{n_north} N, +{n_south} S, +{n_west} W, +{n_east} E cells; "
            f"shape {(nt, y_old.size, x_old.size)} -> {(nt, ny, nx)})"
        )
        logger.info("grew extent (snapshot %s): %s", snapshot, new_edges)

        if verify_samples:
            self._verify_growth(repo, before, snapshot, x_old, y_old,
                                verify_samples, seed)
        return snapshot

    def _verify_growth(self, repo, before: str, after: str, x_old: np.ndarray,
                       y_old: np.ndarray, n: int, seed: int) -> None:
        """Compare ``n`` random pixels of time step 0, by coordinate, across snapshots."""
        rng = np.random.default_rng(seed)
        xi = rng.integers(0, x_old.size, n)
        yi = rng.integers(0, y_old.size, n)
        xs = xr.DataArray(x_old[xi], dims="pt")
        ys = xr.DataArray(y_old[yi], dims="pt")
        old = xr.open_zarr(repo.readonly_session(snapshot_id=before).store,
                           group=self.group_name, consolidated=False, chunks=None)
        new = xr.open_zarr(repo.readonly_session(snapshot_id=after).store,
                           group=self.group_name, consolidated=False, chunks=None)
        a = old[self.variable].isel(time=0).sel(x=xs, y=ys, method="nearest").values
        b = new[self.variable].isel(time=0).sel(x=xs, y=ys, method="nearest").values
        if not np.array_equal(a, b):
            raise RuntimeError(
                f"extent growth verification failed: {int((a != b).sum())}/{n} sampled "
                f"pixels differ between snapshots {before} and {after}")
        logger.info("verified %d sampled pixels unchanged after growth", n)

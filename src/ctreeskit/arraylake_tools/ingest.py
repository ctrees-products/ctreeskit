"""
ingest.py

Annual GeoTIFF/VRT -> Icechunk ingestion connector.

Reads finished per-year raster mosaics (e.g. the CIDDR annual pantropical VRTs that
Ricardo's R pipeline writes to S3) and writes them into an Arraylake/Icechunk repo with
an ``(time, y, x)`` layout, following the CTrees geo-platform design proposal (§4):

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

The GeoTIFF/VRT read + Icechunk write mechanics live here (in ``ctreeskit``) so thin
service wrappers -- e.g. an ECS task in ``web-backend-science`` -- only need to load a
config and call these methods.
"""

# Standard library imports
import json
from importlib import resources
from typing import Any, Dict, Optional, Tuple

# Third-party library imports
import numpy as np
import pandas as pd
import dask.array as da
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)
from icechunk.xarray import to_icechunk
from arraylake import Client as arraylakeClient

# Local application/library specific imports
from .common import ArraylakeDatasetConfig

# Map the config's coarse ``unit_type`` to a concrete numpy dtype when no explicit
# ``dtype`` is given. An explicit ``dtype`` in the variable config always wins so that
# categorical products can preserve their native (often uint8) storage.
_UNIT_TYPE_DTYPE = {"int": "int16", "float": "float32"}


def load_config(name: str) -> Dict[str, Any]:
    """
    Load an *example* dataset-config schema bundled in
    ``ctreeskit/arraylake_tools/datasets_config``.

    Only placeholder/template schemas ship in this public package (e.g.
    ``"categorical_raster_wtih_x_y_time"``). Real dataset configs reference internal
    S3 paths and are kept out of this public repo -- they live in the private
    ``ctrees-products/ctreeskit-internal`` repo. Load those from S3 with
    :class:`~ctreeskit.arraylake_tools.common.ArraylakeDatasetConfig`, or pass the
    parsed dict straight to :class:`AnnualRasterIngester`.

    Parameters
    ----------
    name : str
        Config file stem, e.g. ``"categorical_raster_wtih_x_y_time"`` (with or
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
        Dataset configuration dictionary (e.g. ``ciddr_30m_pantropical.json`` in the
        private ``ctrees-products/ctreeskit-internal`` repo).
    token : Optional[str]
        Arraylake API token. If omitted (and no ``client`` is passed), the client falls
        back to the cached credentials from ``arraylake auth login`` / the
        ``ARRAYLAKE_TOKEN`` environment variable.
    bucket_nickname : str
        Bucket-config nickname used when creating the repo (default ``arraylake-datasets``).
    client : Optional[arraylake.Client]
        An already-constructed Arraylake client to reuse (e.g. the one a service-function
        runner built from ``ARRAYLAKE_TOKEN``). Takes precedence over ``token``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        token: Optional[str] = None,
        bucket_nickname: str = "arraylake-datasets",
        client: Optional[Any] = None,
    ):
        self._configure(config, bucket_nickname)
        # Arraylake connectivity: reuse an injected client, else token, else cached creds.
        self.token = token
        if client is not None:
            self.client = client
        else:
            self.client = arraylakeClient(token=token) if token else arraylakeClient()

    def _configure(self, config: Dict[str, Any], bucket_nickname: str = "arraylake-datasets") -> None:
        """Parse the dataset config into the attributes the connector operates on."""
        self.config = config
        self.bucket_nickname = bucket_nickname

        self.dataset_name = config.get("dataset_name")
        self.organization = config.get("organization", "ctrees")
        self.repo_name = config.get(
            "repo", f"{self.organization}/{self.dataset_name}")
        self.crs = config.get("crs", "EPSG:4326")

        # Resolve the single group + variable this connector operates on.
        self.group_name, group_cfg = self._resolve_group(config)
        self.time_config = group_cfg.get("time")
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
    def _resolve_group(config: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return the (name, config) of the single group this connector ingests."""
        groups = config.get("groups", {})
        if not groups:
            raise ValueError("config has no 'groups' entry to ingest")
        if len(groups) > 1:
            raise ValueError(
                "AnnualRasterIngester expects exactly one group; found "
                f"{sorted(groups)}. Ingest groups individually.")
        name = next(iter(groups))
        return name, groups[name]

    @staticmethod
    def _resolve_variable(group_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return the (name, config) of the single data variable in a group."""
        variables = {k: v for k, v in group_cfg.items() if k != "time"}
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

    def create_repo(self, exist_ok: bool = True) -> None:
        """
        Get or create the Icechunk repo on Arraylake.

        Uses the client's ``get_or_create_repo`` rather than a manual existence probe
        + ``create_repo`` -- it already handles the create-if-missing logic
        atomically, so this method doesn't need to distinguish a "not found" error
        from an "already exists" one itself.

        Parameters
        ----------
        exist_ok : bool
            If True (default), silently get-or-create. If False, raise when the repo
            already exists instead.
        """
        try:
            self.client.get_repo(self.repo_name)
            exists = True
        except Exception:
            exists = False

        if exists and not exist_ok:
            raise ValueError(f"Repository already exists: {self.repo_name}")

        print(
            f"Repository already exists: {self.repo_name}" if exists
            else f"Creating repository: {self.repo_name} (bucket={self.bucket_nickname})"
        )
        self.client.get_or_create_repo(
            name=self.repo_name,
            bucket_config_nickname=self.bucket_nickname,
        )

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
            If True, use ``mode="w"`` (replace an existing array); otherwise ``mode="w-"``.

        Returns
        -------
        str
            The snapshot id of the initialization commit.
        """
        years = self._years()
        if not years:
            raise ValueError("time config is required to initialize an annual schema")
        template_year = template_year or years[0]

        # Read only the grid from the template mosaic (coordinates are derived from the
        # geotransform, so this does not read pixel data).
        template = rioxarray.open_rasterio(self._year_uri(template_year), masked=False)
        template = template.squeeze("band", drop=True)
        x = template.x.values
        y = template.y.values
        template.close()

        time = pd.date_range(
            start=self.time_config["start"],
            end=self.time_config["end"],
            freq=self.time_config.get("freq", "YS"),
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

        session = self._repo().writable_session("main")
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
        print(f"initialized schema (snapshot {snapshot}) shape={shape} dtype={self.dtype}")
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
        if bbox is None:
            da_year = rioxarray.open_rasterio(
                self._year_uri(year), masked=False,
                chunks={"band": 1, "y": ychunk, "x": xchunk}, lock=False,
            ).squeeze("band", drop=True)
        else:
            da_year = rioxarray.open_rasterio(
                self._year_uri(year), masked=False,
            ).squeeze("band", drop=True)
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
            repo.readonly_session("main").store, group=self.group_name,
            consolidated=False, chunks=None,
        )
        stored_years = pd.to_datetime(stored.time.values).year.tolist()
        target = pd.Timestamp(f"{year}-01-01")

        session = repo.writable_session("main")
        if year in stored_years:
            t_idx = stored_years.index(year)
            region = {"time": slice(t_idx, t_idx + 1)}
            if bbox is None:
                region["y"] = slice(None)
                region["x"] = slice(None)
            else:
                region["y"] = self._index_slice(stored.y.values, ds_year.y.values)
                region["x"] = self._index_slice(stored.x.values, ds_year.x.values)
            # Full-extent writes cover whole storage chunks, so keep the safe-chunk
            # guard (needed for distributed writes). A bbox window may end mid-chunk;
            # that is a single-writer verification path, so relax the guard.
            to_icechunk(
                ds_year, session, group=self.group_name, region=region,
                align_chunks=True, safe_chunks=(bbox is None),
            )
            action = f"region {region}"
        elif target > stored.time.values.max():
            if bbox is not None:
                raise ValueError(
                    "appending a new year must cover the full extent; drop bbox for "
                    f"year {year} (beyond the initialized axis)")
            to_icechunk(ds_year, session, group=self.group_name, append_dim="time")
            action = "append time"
        else:
            raise ValueError(
                f"year {year} is before the initialized axis and not a slot; "
                "re-initialize the schema to include it")

        snapshot = session.commit(f"ingest {self.variable} {year} ({action})")
        print(f"ingested {year}: {action} -> snapshot {snapshot}")
        if tag:
            repo.create_tag(tag, snapshot)
            print(f"tagged snapshot {snapshot} as '{tag}'")
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
                "bbox subset does not align with the stored grid "
                f"(first coord {subset_coord[0]!r} vs nearest stored "
                f"{stored_coord[start]!r}, tolerance {tol!r}). Snap the bbox "
                "to the stored grid (same resolution and origin) before a "
                "region write.")
        return slice(start, stop)

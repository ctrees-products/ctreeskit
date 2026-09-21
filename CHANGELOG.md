# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

Versions prior to 0.2.0 were not tracked in this changelog.

## [Unreleased]

## [0.3.0] - 2026-09-21

### Added
- `AnnualRasterIngester.grow_extent(extent=None)` enlarges a stored `(time, y, x)`
  domain without rewriting pixels: zarr resize plus Icechunk `shift_array` for
  north/west growth (snapped outward to whole chunks), coordinate arrays and
  `GeoTransform` rewritten, sampled verification against the pre-growth snapshot.
  The target defaults to the group's configured `extent`, so the workflow is to
  update the config first and then grow the stored domain to match.
- Optional per-group `extent` `[minx, miny, maxx, maxy]` (outer edges) in dataset
  configs sizes the grid at `initialize_schema`; the template raster must sit on
  that lattice.
- `AnnualRasterIngester(group=...)` selects a group from multi-group configs.
- `xr_analyzer.xr_vectorize_module`: turns categorical change layers into vector
  patches and events. `patches_from_mask`, `patches_from_categorical` and
  `patches_over_time` polygonize connected pixel regions into a GeoDataFrame
  carrying `patch_id`, `pixel_count` and true `area_ha` (per-cell geodesic or
  equal-area measurement on lat/lon grids, transform-derived on metre-based
  projected grids), with a minimum-mapping-unit filter applied to measured area.
  `merge_patches` groups patches within a metric gap into events — buffering in a
  projected CRS, never in degrees — and returns the events plus a membership
  table. `assign_to_polygons` produces a many-to-many overlap table against
  reference polygons. `write_geoparquet` / `read_geoparquet` round-trip either
  table as GeoParquet 1.0.0 with the CRS recorded.
- Optional `vector` extra (geopandas, pyarrow) backing the vector output. The
  module imports without it; the functions that need it raise `ImportError` with
  install instructions.
- `xr_analyzer.xr_observations_module`: one observation record for detection
  layers from unlike sources. `OBSERVATION_FIELDS` fixes the normalized
  variables and dtypes — state (`STATE_CODES`), first/confirm/last observation
  dates, look and positive counts, native and rescaled confidence, class hint
  and date precision (`DATE_PRECISION`) — with `source`, `sensor`,
  `pixel_size_m`, `snapshot_id`, `read_at`, `has_counts` and the CRS on the
  dataset. Three adapters read onto it, always on the source grid and never
  resampling: `observations_from_dated_codes` decodes a 2-D raster of dated
  state codes chunk by chunk on dask-backed input;
  `observations_from_annual_alert_days` decodes annual slices carrying a
  per-pixel day value, keeping the `time` dimension because a pixel may alert in
  more than one year, with `select_month` selecting one calendar month; and
  `observations_from_points` brings a point table on directly, rescaling
  categorical or numeric confidence to [0, 1], with `points_to_grid_mask`
  rasterizing it onto a reference grid.
- `observations_to_table` flattens a record to one row per detected pixel —
  identical columns and dtypes whatever the adapter, so records concatenate —
  and `write_observations` / `read_observations` round-trip it as Parquet.
  `observations_mask` reduces a record to the boolean mask `patches_from_mask`
  takes, optionally filtered to one month, joining observations to patches and
  events.
- `observations_from_tier_steps` normalizes a regularly stepped categorical
  layer that writes each pixel at one step only: `tier_map` routes the layer's
  own tier vocabulary onto the shared states, the step coordinate supplies the
  date, and the step dimension collapses to a 2-D record. A pixel written at
  more than one step keeps its latest write and is flagged `multi_step`.
  `observations_from_tier_steps_by_step` keeps the steps for per-step masks,
  with `select_step_month` selecting one calendar month.

### Changed
- `ingest_year` places every source on the stored grid by coordinate. A source
  covering part of the domain fills only its window; one that is off-lattice, at
  another resolution, or larger than the domain raises `ValueError` instead of
  failing on shape (or, for same-shape sources with a different origin, writing
  to the wrong place). **Breaking** only for callers that relied on that
  unchecked full-extent write.
- The package type-checks cleanly under mypy, and CI enforces it. Geometry inputs
  to `process_geometry`, `clip_ds_to_geom`, and `create_proportion_geom_mask` are
  now typed (and validated) as Shapely `BaseGeometry` objects; other duck-typed
  geometry objects are rejected with a `ValueError` up front.
- Removed unused core dependencies `scipy`, `cf-xarray`, and `python-dotenv`;
  `geopandas` moved from core to the `zonal` extra (floor raised to `>=1.0`).
  **Breaking** for consumers that relied on these installing transitively —
  declare them directly instead.
- Removed the `interactive` extra (`ipykernel`, `ipyleaflet`). **Breaking** for
  consumers installing `ctreeskit[interactive]`.
- Core dependency floors raised to currently supported releases: `numpy>=2.0`,
  `pandas>=2.2.2`, `shapely>=2.0.6`, `rioxarray>=0.17.0`.
- S3 access moved behind a new optional `s3` extra backed by boto3; `s3fs` is no
  longer a dependency. Loading dataset configs or `s3://` GeoJSON paths raises
  `ImportError` with install instructions unless `ctreeskit[s3]` is installed;
  constructing `ArraylakeDatasetConfig` no longer requires any AWS library.
  **Breaking** for consumers reading from S3 (install the extra) or relying on
  `s3fs` transitively.
- The `zonal` extra installs on all supported Python versions (3.12–3.14): the
  `sparse` dependency lost its Python upper-bound marker and a `numba>=0.63`
  floor guarantees a Python-3.14-capable numba. numba caps numpy below 2.5, so
  environments with this extra resolve numpy to a 2.4.x release.

## [0.2.0]

### Added
- `AnnualRasterIngester` (`arraylake_tools.ingest`): annual GeoTIFF/VRT -> Icechunk
  ingestion connector with a `(time, y, x)` layout. Separates one-time schema
  allocation (`initialize_schema`) from per-year population (`ingest_year`), which
  writes into a pre-allocated time slot as a region write or appends a new year.
  Supports an optional spatial `bbox` for fast verification, an Icechunk `branch_name`
  for testing off the production branch, and per-year tagging.

### Changed
- Dependency floors raised for icechunk (`>=2.0.6`), arraylake (`>=1.1.1`), and zarr
  (`>=3.1.0`); `requires-python>=3.12`. **Breaking** for consumers on older
  icechunk/arraylake/zarr or Python < 3.12.
- Migrated project tooling to [uv](https://docs.astral.sh/uv/): `pyproject.toml` +
  committed `uv.lock` replace `requirements.txt`; CI uses `astral-sh/setup-uv`.
  Dev tooling moved into a PEP 735 dependency group.
- The heavy, GDAL-dependent `dask_analyzer` subpackage (exactextract, odc-geo) moved
  behind an optional `zonal` extra with a guarded import, so the core package stays
  pip-installable with no GDAL requirement.
- Categorical zonal stats are computed with a single flag-aware groupby
  ([flox](https://flox.readthedocs.io/)-accelerated, dask-compatible); flox is a new
  core dependency. CF `flag_values` metadata, when present, defines the class label
  set, and `calculate_categorical_area_stats(reshape=False)` returns a tidy
  long-format DataFrame.
- `create_combined_classification` encodes class pairs as integers
  (`primary * modulus + secondary`, with the modulus stored in the
  `combined_modulus` attribute). **Breaking** for consumers decoding the previous
  decimal-fraction codes.
- CI runs `ruff check` and triggers on pull requests as well as pushes.
- Bundled config template filenames are spelled correctly:
  `categorical_raster_with_x_y[_time].json` and
  `continuous_raster_with_x_y[_time].json`. **Breaking** for callers passing the old
  misspelled stems (`wtih`, `conitnous`, `continous`) to `load_config`.
- `__version__` is single-sourced from the package metadata
  (`importlib.metadata.version`); the per-subpackage version strings are gone.
- `AnnualRasterIngester` reports progress through a module logger
  (`logging.getLogger("ctreeskit.arraylake_tools.ingest")`) instead of `print`, and
  config-loading errors chain the original exception (`raise ... from e`).
- `AnnualRasterIngester` no longer falls back to a default Arraylake organization:
  dataset configs must carry `organization` (or a full `repo` name). **Breaking**
  for configs that relied on the implicit default.

### Removed
- The `dask_analyzer` subpackage (`calculate_categorical_area_stats_dask`,
  `create_area_ds_from_degrees_ds_dask`, `reproject_match_dask`, `geometry_clip_rio`).
  The core zonal-stats functions are now dask-compatible themselves (flox-backed
  groupby), and coverage-fraction rasterization comes from the released
  [rasterix](https://pypi.org/project/rasterix/) package, which the `zonal` extra now
  installs — fully pip-installable, no GDAL required (see the coverage-fraction
  example in `docs/xr_analyzer.md`). **Breaking** for consumers importing the
  `_dask` variants.
- The legacy `arraylake_tools` classes `ArraylakeRepoCreator`, `ArraylakeRepoInitializer`,
  and `ArraylakeRepoPopulator`. They were incompatible with the icechunk 2.x /
  arraylake 1.x APIs this package now requires; `AnnualRasterIngester` covers the
  create/initialize/populate workflow with current Icechunk semantics. **Breaking**
  for consumers importing these classes.

### Fixed
- Zonal-stats functions no longer crash on documented input shapes (e.g. `area_ds`
  passed as a `DataArray`).
- Static (non-time) categorical area stats reported only the first class and dropped
  the rest; all classes are now reported, and the static and time-series paths agree
  on the same data.
- Class columns are matched to `flag_meanings` by flag value; non-contiguous class
  codes (e.g. 10/20/30) were previously mislabeled or dropped by the positional
  mapping.
- Combined-classification codes are collision-free for class values >= 10 (e.g.
  primary 3 / secondary 12 no longer merges with primary 4 / secondary 2).
- The automatic area-method choice in `create_area_ds_from_degrees_ds` considers the
  whole latitude axis (geodesic when any latitude is poleward of 70°); previously it
  inspected only the first row, so the method flipped with storage orientation.
- `create_proportion_geom_mask` works on all paths: the default path no longer
  crashes, the below-threshold binary fallback no longer swaps the clip arguments,
  and weighted proportions are computed on the clipped grid from the geometry
  footprint (previously misregistered whenever clipping cropped the raster, with
  zero-valued pixels skipped).
- GeoJSON input is accepted per RFC 7946 even without a `crs` member.
- Bounding-box region writes in `AnnualRasterIngester` are rejected when the
  requested subset doesn't align with the stored grid, instead of silently writing
  pixels shifted by up to half a cell.

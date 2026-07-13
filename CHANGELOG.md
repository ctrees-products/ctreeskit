# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/).

Versions prior to 0.2.0 were not tracked in this changelog.

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
- `create_proportion_geom_mask` works on all paths: the default path no longer
  crashes, the below-threshold binary fallback no longer swaps the clip arguments,
  and weighted proportions are computed on the clipped grid from the geometry
  footprint (previously misregistered whenever clipping cropped the raster, with
  zero-valued pixels skipped).
- GeoJSON input is accepted per RFC 7946 even without a `crs` member.
- Bounding-box region writes in `AnnualRasterIngester` are rejected when the
  requested subset doesn't align with the stored grid, instead of silently writing
  pixels shifted by up to half a cell.

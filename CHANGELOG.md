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

### Fixed
- Zonal-stats functions no longer crash on documented input shapes (e.g. `area_ds`
  passed as a `DataArray`).
- GeoJSON input is accepted per RFC 7946 even without a `crs` member.
- Bounding-box region writes in `AnnualRasterIngester` are rejected when the
  requested subset doesn't align with the stored grid, instead of silently writing
  pixels shifted by up to half a cell.

# Installation

`ctreeskit` requires **Python 3.12 or newer**. The core package is pure-Python to
install (no GDAL, no cloud SDK) and works with either `pip` or
[uv](https://docs.astral.sh/uv/).

## Core install

The package is distributed from its git repository:

```bash
pip install "ctreeskit @ git+https://github.com/ctrees-products/ctreeskit.git"
```

To pin a released version, append a tag:

```bash
pip install "ctreeskit @ git+https://github.com/ctrees-products/ctreeskit.git@v1.0.0"
```

## Optional extras

Two features live behind optional extras so the core install stays lightweight.
Combine them in the usual way, e.g. `ctreeskit[zonal,s3]`.

(zonal-extra)=
### `zonal` — coverage-fraction weighted statistics

Adds [rasterix](https://rasterix.readthedocs.io/) (exactextract-backed coverage
grids), `geopandas`, and their dependencies for geometry-weighted zonal statistics:

```bash
pip install "ctreeskit[zonal] @ git+https://github.com/ctrees-products/ctreeskit.git"
```

This extra pulls in `numba`, which caps `numpy` below 2.5, so environments with the
`zonal` extra resolve `numpy` to a 2.4.x release. It installs on all supported Python
versions (3.12–3.14).

(s3-extra)=
### `s3` — reading from `s3://` paths

Adds `boto3` for loading dataset configs and GeoJSON geometries directly from
`s3://` URIs, using the standard AWS credential chain:

```bash
pip install "ctreeskit[s3] @ git+https://github.com/ctrees-products/ctreeskit.git"
```

Without this extra, local file paths work everywhere; only reading from `s3://`
raises an `ImportError` that points back here.

## Development install with uv

Clone the repository and sync the locked environment:

```bash
git clone https://github.com/ctrees-products/ctreeskit.git
cd ctreeskit

# Core + dev tooling (pip-only, no GDAL)
uv sync --group dev

# Add the optional extras when working on those features
uv sync --extra zonal --extra s3 --group dev
```

Run the test suite, linters, and type checker with:

```bash
uv run pytest
uv run ruff check .
uv run mypy src/ctreeskit
```

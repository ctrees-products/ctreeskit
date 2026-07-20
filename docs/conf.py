"""Sphinx configuration for the ctreeskit documentation."""

from importlib.metadata import version as _pkg_version

# -- Project information -----------------------------------------------------

project = "ctreeskit"
copyright = "CTrees"
author = "CTrees"

# Full version, single-sourced from the installed package metadata.
release = _pkg_version("ctreeskit")
# Short X.Y version.
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST (Markdown) ---------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
]
myst_heading_anchors = 3

# -- Autodoc / napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
# The codebase mixes numpy- and google-style docstrings; parse both.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Render class Attributes as inline :ivar: fields so they are not also emitted
# as separate attribute descriptions (which would duplicate the annotations).
napoleon_use_ivar = True
always_document_param_types = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "rioxarray": ("https://corteva.github.io/rioxarray/stable", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/stable", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "shapely": ("https://shapely.readthedocs.io/en/stable", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable", None),
    "dask": ("https://docs.dask.org/en/stable", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"ctreeskit {version}"
html_static_path = ["_static"]

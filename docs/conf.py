# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys
from pathlib import Path

two_up = Path(__file__).parents[1]
sys.path.insert(0, two_up.as_posix())
# -- Project information -----------------------------------------------------

project = 'dyntapy'
copyright = '2022, Paul Ortmann'
author = 'Paul Ortmann'

# The full version, including alpha/beta/rc tags
release = '0.2.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon','sphinx.ext.autosummary'
              ]

napoleon_google_docstring = False  # using numpydoc via napoleon rather than directly
# numpydoc itself is rather strict

napoleon_preprocess_types = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

# adding mock imports needed for any machine to build the docs for dyntapy using autodoc
autodoc_mock_imports = ['networkx', 'osmnx', 'shapely', 'numba', 'bokeh', 'numpy',
                        'scipy', 'geojson', 'pandas', 'geopandas','matplotlib',
                        'pyproj']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

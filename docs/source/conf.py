# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

project = "matlap"
copyright = "2026, David Knowles"
author = "David Knowles"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

autodoc_typehints = "signature"
autodoc_member_order = "bysource"
autodoc_mock_imports = ["jax", "jaxlib", "jax.numpy", "jax.scipy", "numpyro", "optax", "numpy"]

suppress_warnings = ["ref.duplicate_object"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = []
language = "en"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

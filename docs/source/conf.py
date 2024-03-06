# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import datetime
from pathlib import Path
from importlib import metadata

sys.path.insert(0, os.path.abspath("../../"))  # Adjust the path as needed

project = "pyaging"
copyright = "2023, Lucas Paulo de Lima Camillo"
author = "Lucas Paulo de Lima Camillo"
from pyaging import __version__

release = version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    # "nbsphinx_link",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",  # needs to be after napoleon
    "sphinx_issues",
    "sphinx_design",
    "scanpydoc",  # needs to be before linkcode
    "sphinx.ext.linkcode",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.imgmath",
    "sphinx.ext.extlinks",
]

templates_path = ["../_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "tutorials/notebooks/*.rst",
]
html_static_path = ["../_static"]
source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_theme_options = dict(
    use_repository_button=True,
    repository_url="https://github.com/rsinghlab/pyaging",
    repository_branch="main",
    navigation_with_keys=False,  # https://github.com/pydata/pydata-sphinx-theme/issues/1492
)
html_logo = "../_static/logo.png"
html_css_files = ["custom.css"]

# -- Options for nbshpinx ----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.0/configure.html

nbsphinx_execute = "never"

# -- Additional configurations from the provided conf.py ----------------------

# ... (Add other configurations from the provided conf.py here)
# Make sure to resolve any conflicts with the existing settings above.

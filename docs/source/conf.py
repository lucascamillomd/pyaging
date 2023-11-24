# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Adjust the path as needed

project = 'pyaging'
copyright = '2023, Lucas Paulo de Lima Camillo'
author = 'Lucas Paulo de Lima Camillo'
release = '0.0.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
    # "myst_nb",
]
templates_path = ['../_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['../_static']
html_logo = 'https://pyaging.readthedocs.io/en/latest/_static/logo.png'
html_css_files = [
    'custom.css',
]

# Change the URL for the "Edit on GitHub" button
html_context = {
    "display_github": True,
    "github_user": "lcamillo",
    "github_repo": "https://github.com/rsinghlab/pyaging",
    "github_version": "main",  # Or specify a branch/tag if needed
    "conf_py_path": "/docs/source/",  # The path to your documentation source
}

# -- Options for nbshpinx ----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.0/configure.html

nbsphinx_execute = "never"
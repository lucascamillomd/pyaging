# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

project = "pyaging"
copyright = "2023, Lucas Paulo de Lima Camillo"
author = "Lucas Paulo de Lima Camillo"

release = version = metadata.version("pyaging")

# -- Copy canonical notebooks into the docs tree at build time ----------------

_conf_dir = Path(__file__).resolve().parent
_repo_root = _conf_dir.parents[1]


def _sync_notebooks():
    for src_dir, dest_dir in (
        (_repo_root / "tutorials", _conf_dir / "tutorials"),
        (_repo_root / "clocks" / "notebooks", _conf_dir / "clock_notebooks"),
    ):
        dest_dir.mkdir(exist_ok=True)
        for stale in sorted(dest_dir.glob("*.ipynb")):
            if not (src_dir / stale.name).is_file():
                stale.unlink()
        for src in sorted(src_dir.glob("*.ipynb")):
            dest = dest_dir / src.name
            if not dest.is_file() or src.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy(src, dest)


_sync_notebooks()

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
    "sphinx.ext.extlinks",
]

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

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/lucascamillomd/pyaging",
    "icon_links": [
        {"name": "PyPI", "url": "https://pypi.org/project/pyaging/", "icon": "fa-brands fa-python"},
        {"name": "Paper", "url": "https://doi.org/10.1093/bioinformatics/btae200", "icon": "fa-solid fa-book-open"},
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "show_prev_next": False,
    "navigation_with_keys": False,
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
    "header_links_before_dropdown": 6,
}
html_context = {
    "default_mode": "auto",
    "github_user": "lucascamillomd",
    "github_repo": "pyaging",
    "github_version": "main",
}
# The Clock Catalogue owns the full width — drop its left section-nav sidebar.
html_sidebars = {"clock_glossary": []}
html_logo = "../_static/logo.png"
html_css_files = [
    "https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700;12..96,800&display=swap",
    "custom.css",
    "clock_explorer.css",
]
html_js_files = ["clock_explorer_core.js", "clock_explorer.js"]

# -- Options for nbshpinx ----------------------------------------------------
# https://nbsphinx.readthedocs.io/en/0.8.0/configure.html

nbsphinx_execute = "never"
suppress_warnings = ["nbsphinx.ipywidgets"]

# -- Generate Clock Explorer data at build time (local + Read the Docs) -------


def _generate_clock_data(app):
    # Ensure this conf dir is importable when builder-inited fires (Read the Docs
    # does not keep the confdir on sys.path by the time the event runs).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from make_clock_data import generate

        local_metadata = Path(__file__).resolve().parents[2] / "clocks" / "metadata" / "all_clock_metadata.pt"
        n = generate(metadata_path=local_metadata) if local_metadata.is_file() else generate()
        logger.info("[clocks] regenerated clocks.json with %s clocks", n)
    except Exception as exc:  # noqa: BLE001 — never break the build
        logger.warning("[clocks] using committed clocks.json (%s)", exc)


def setup(app):
    app.connect("builder-inited", _generate_clock_data)

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

html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/lucascamillomd/pyaging",
    "source_branch": "main",
    "source_directory": "docs/source",
    "light_css_variables": {
        "color-brand-primary": "#2f6394",
        "color-brand-content": "#2f6394",
    },
    "dark_css_variables": {
        "color-brand-primary": "#7fb0e8",
        "color-brand-content": "#7fb0e8",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/lucascamillomd/pyaging",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0"'
                ' viewBox="0 0 16 16" height="1.2em" width="1.2em">'
                '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17'
                ".55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-"
                ".82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 "
                "2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82"
                "-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68"
                " 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56."
                "82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01"
                ' 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>'
                "</svg>"
            ),
            "class": "",
        },
    ],
}
# scanpydoc.rtd_github_links resolves source links from these
html_context = {
    "github_user": "lucascamillomd",
    "github_repo": "pyaging",
    "github_version": "main",
}
html_logo = "../_static/logo.png"
html_favicon = "../_static/logo.png"
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

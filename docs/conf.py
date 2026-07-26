from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "hnsbi-toolkit"
author = "hnsbi-toolkit contributors"
copyright = "2026, hnsbi-toolkit contributors"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]
myst_heading_anchors = 3

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

html_theme = "furo"
html_title = "hnsbi-toolkit"
html_logo = "_static/hnsbi-logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": True,
}

nitpicky = False

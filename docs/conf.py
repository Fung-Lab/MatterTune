from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "MatterTune"
copyright = "2024, MatterTune Team"
author = "MatterTune Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",
]

# MyST Markdown settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "tasklist",
]

# Theme settings
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = None
html_favicon = None

# General settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# AutoDoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Add autosummary settings
autosummary_generate = True
autosummary_imported_members = True
templates_path = ["_templates"]

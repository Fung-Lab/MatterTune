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
    "exclude-members": "__weakref__, __pydantic_core_schema__, __pydantic_validator__, __pydantic_serializer__, \
__pydantic_fields_set__, __pydantic_extra__, __pydantic_private__, __pydantic_post_init__, __pydantic_decorators__, \
__pydantic_parent_namespace__, __pydantic_generic_metadata__, __pydantic_custom_init__, __pydantic_complete__, \
__fields__, __fields_set__, model_fields, model_config, model_computed_fields, __class_vars__, __private_attributes__, \
__signature__, __pydantic_root_model__, __slots__, __dict__, model_extra, model_fields_set, model_post_init",
    "show-module-summary": True,
}


# To exclude private members (those starting with _)
def skip_private_members(app, what, name, obj, skip, options):
    if (
        name.startswith("_")
        and not name.startswith("__")
        and not name.startswith("__pydantic")
    ):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_private_members)


autodoc_mock_imports = [
    # "torch",
    "pytorch_lightning",
    "torch-geometric",
    "torchmetrics",
]

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Enable automatic doc generation
autosummary_generate = True
autodoc_member_order = "bysource"
add_module_names = True

# Custom templates
autodoc_template_dir = ["_templates"]
templates_path = ["_templates"]

# always_use_bars_union for type hints
always_use_bars_union = True
always_document_param_types = True

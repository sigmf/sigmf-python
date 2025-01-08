# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration file for the Sphinx documentation builder."""

import datetime
import re
from pathlib import Path

# parse info from project files

root = Path(__file__).parent.parent.parent
with open(root / "sigmf" / "__init__.py", "r") as handle:
    init = handle.read()
    toolversion = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', init).group(1)
    specversion = re.search(r'__specification__\s*=\s*[\'"]([^\'"]*)[\'"]', init).group(1)
print("DBUG", toolversion, specversion)

# -- Project information

project = "sigmf"
author = "Multiple Authors"
copyright = f"2017-{datetime.date.today().year}, {author}"

release = toolversion
version = toolversion

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"

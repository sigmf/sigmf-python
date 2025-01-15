# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Configuration file for the Sphinx documentation builder."""

import datetime
import re
import sys
from pathlib import Path

# parse info from project files

root = Path(__file__).parent.parent.parent
with open(root / "sigmf" / "__init__.py", "r") as handle:
    init = handle.read()
    toolversion = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', init).group(1)
    specversion = re.search(r'__specification__\s*=\s*[\'"]([^\'"]*)[\'"]', init).group(1)

# autodoc needs special pathing
sys.path.append(str(root))

# -- Project information

project = "sigmf"
author = "Multiple Authors"
copyright = f"2017-{datetime.date.today().year}, {author}"

release = toolversion
version = toolversion

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # allows numpy-style docstrings
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
html_favicon = "https://raw.githubusercontent.com/wiki/sigmf/SigMF/logo/logo-icon-32-folder.png"
html_logo = "https://raw.githubusercontent.com/sigmf/SigMF/refs/heads/main/logo/sigmf_logo.svg"

# -- Options for EPUB output

epub_show_urls = "footnote"

# Method to use variables within rst files
# https://stackoverflow.com/a/69211912/760099

variables_to_export = [
    "toolversion",
    "specversion",
]
frozen_locals = dict(locals())
rst_epilog = '\n'.join(map(lambda x: f".. |{x}| replace:: {frozen_locals[x]}", variables_to_export))
del frozen_locals
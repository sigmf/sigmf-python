# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Schema IO"""

import json
from pathlib import Path

from . import __version__ as toolversion

SCHEMA_META = "schema-meta.json"
SCHEMA_COLLECTION = "schema-collection.json"


def get_schema(version=toolversion, schema_file=SCHEMA_META):
    """
    Load JSON Schema to for either a `sigmf-meta` or `sigmf-collection`.

    TODO: In the future load specific schema versions.
    """
    schema_dir = Path(__file__).parent
    with open(schema_dir / schema_file, "rb") as handle:
        schema = json.load(handle)
    return schema

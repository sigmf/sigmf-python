# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Schema IO"""

import json
from pathlib import Path

from . import utils

SCHEMA_META = "schema-meta.json"
SCHEMA_COLLECTION = "schema-collection.json"


def get_schema(version=None, schema_file=SCHEMA_META):
    """
    Load JSON Schema to for either a `sigmf-meta` or `sigmf-collection`.

    TODO: In the future load specific schema versions.
    '''

    schema_path = Path.joinpath(utils.get_schema_path(Path(utils.__file__).parent), schema_file)
    with open(schema_path, 'rb') as handle:
        schema = json.load(handle)
    return schema

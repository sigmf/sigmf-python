# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Global pydantic Field types, including DOI strings, version strings etc."""

from typing_extensions import Annotated
from pydantic import StringConstraints, Field


# version string type of pattern e.g "1.0.0"
VERSION_STR = Annotated[str, StringConstraints(strip_whitespace=True, pattern="[0-9]+.[0-9]+.[0-9]+(dev)?")]
# regex pattern for data type strings.
DATATYPE_STR = Annotated[str, StringConstraints(strip_whitespace=True, pattern="[cr][fui](8|16|32|64)(_[bl]e)?")]
# DOI string as defined by ISO 26324
DOI_STR = Annotated[str, StringConstraints(pattern="(doi:)?[a-zA-Z0-9_.]+/[a-zA-Z0-9_.]+")]
# frequency type with range
FREQUENCY_TYPE = Annotated[
    float, Field(ge=-1000000000000.0, le=1000000000000.0, allow_inf_nan=False, description="Frequency in Hz")
]

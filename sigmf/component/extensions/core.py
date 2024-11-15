# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core:extension definition for SigMFGlobal in Pydantic."""

from pydantic import BaseModel, ConfigDict, Field

from sigmf.component.pydantic_types import VERSION_STR


class SigMFCoreExtension(BaseModel):
    """core:extensions to a Global Object."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="The name of the SigMF extension namespace.", frozen=True)
    version: VERSION_STR = Field(
        ..., description="The version of the extension namespace specification used.", examples=["1.0.0"], frozen=True
    )
    optional: bool = Field(
        ..., description="If this field is `true`, the extension is REQUIRED to parse this Recording.", frozen=True
    )

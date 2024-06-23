# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
"""GEOJson Definitions in Pydantic."""

from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from pydantic import (
    BaseModel,
    Field,
    AliasPath,
    AliasChoices,
    model_serializer,
    model_validator,
)


class WGS84Coordinate(BaseModel):
    """Defines latitude, longitude and altitude using the WGS84 coordinate reference system."""

    latitude: float = Field(
        ...,
        description="latitude in decimal degrees",
        validation_alias=AliasChoices("latitude", AliasPath("coordinates", 0)),
    )
    longitude: float = Field(
        ...,
        description="longitude in decimal degrees",
        validation_alias=AliasChoices("longitude", AliasPath("coordinates", 1)),
    )
    altitude: Optional[float] = Field(
        None,
        description="in meters above the WGS84 ellipsoid",
        validation_alias=AliasChoices("altitude", AliasPath("coordinates", 2)),
    )

    @model_serializer
    def ser_model(self) -> List[float]:
        """serialize the whole model into a list of floats"""
        if self.altitude:
            return [self.latitude, self.longitude, self.altitude]
        return [self.latitude, self.longitude]


class GeoJSONPoint(BaseModel):
    """GeoJsonPoint object as defined by RFC 7946."""

    # for Python 3.8+ - replace "type" with "Literal['point']"
    type_: Literal["Point"] = Field(
        "Point", alias="type", description="Type of GeoJSON object as per RFC 5870", frozen=True
    )
    coordinates: WGS84Coordinate = Field(..., description="WGS84 coordinate reference system.")

    @model_validator(mode="before")
    @classmethod
    def convert_coordinate_to_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Converts {"coordinates": [x, y, _z]} to {"coordinates": {"coordinates": [x, y, _z]}}"""
        # check that the length is >=2 and < 4
        if "coordinates" not in data:
            raise KeyError("key `coordinates` not found.")
        if not 1 < len(data["coordinates"]) < 4:
            raise ValueError(f"`coordinates` length must be 2 or 3 (lat, lon, <alt>), not {len(data['coordinates'])}.")
        data["coordinates"] = {"coordinates": data["coordinates"]}
        return data

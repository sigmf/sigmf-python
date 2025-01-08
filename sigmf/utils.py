# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Utilities"""

import re
import sys
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from . import error

SIGMF_DATETIME_ISO8601_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def get_sigmf_iso8601_datetime_now() -> str:
    """Get current UTC time as iso8601 string."""
    return datetime.now(timezone.utc).strftime(SIGMF_DATETIME_ISO8601_FMT)


def parse_iso8601_datetime(string: str) -> datetime:
    """
    Parse an iso8601 string as a datetime struct.
    Input string (indicated by final Z) is in UTC tz.

    Example
    -------
    >>> parse_iso8601_datetime("1955-11-05T06:15:00Z")
    datetime.datetime(1955, 11, 5, 6, 15, tzinfo=datetime.timezone.utc)
    """
    match = re.match(r"^(?P<dt>.*)(?P<frac>\.[0-9]{7,})Z$", string)
    if match:
        # string exceeds max precision allowed by strptime -> truncate to µs
        groups = match.groupdict()
        length = min(7, len(groups["frac"]))
        string = "".join([groups["dt"], groups["frac"][:length], "Z"])

    if "." in string:
        # parse float seconds
        format_str = SIGMF_DATETIME_ISO8601_FMT
    else:
        # parse whole seconds
        format_str = SIGMF_DATETIME_ISO8601_FMT.replace(".%f", "")
    return datetime.strptime(string, format_str).replace(tzinfo=timezone.utc)


def dict_merge(a_dict: dict, b_dict: dict) -> dict:
    """
    Recursively merge `b_dict` into `a_dict`.
    `b_dict[key]` will overwrite `a_dict[key]` if it exists.

    Example
    -------
    >>> a, b = {0:0, 1:2}, {1:3, 2:4}
    >>> dict_merge(a, b)
    {0: 0, 1: 3, 2: 4}
    """
    if not isinstance(b_dict, dict):
        return b_dict
    result = deepcopy(a_dict)
    for key, value in b_dict.items():
        if key in result and isinstance(result[key], dict):
            result[key] = dict_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def get_schema_path(module_path: str) -> str:
    """
    TODO: Allow getting different schemas for specific SigMF versions
    """
    return module_path


def get_endian_str(ray: np.ndarray) -> str:
    """Return SigMF compatible endianness string for a numpy array"""
    if not isinstance(ray, np.ndarray):
        raise error.SigMFError("Argument must be a numpy array")
    atype = ray.dtype

    if atype.byteorder == "<":
        return "_le"
    if atype.byteorder == ">":
        return "_be"
    # endianness is then either '=' (native) or '|' (doesn't matter)
    return "_le" if sys.byteorder == "little" else "_be"


def get_data_type_str(ray: np.ndarray) -> str:
    """
    Return the SigMF datatype string for the datatype of numpy array `ray`.

    NOTE: this function only supports native numpy types so interleaved complex
    integer types are not supported.
    """
    if not isinstance(ray, np.ndarray):
        raise error.SigMFError("Argument must be a numpy array")
    atype = ray.dtype
    if atype.kind not in ("u", "i", "f", "c"):
        raise error.SigMFError("Unsupported data type:", atype)
    data_type_str = ""
    if atype.kind == "c":
        data_type_str += "cf"
        # units are component bits, numpy complex types len(I)+len(Q)
        data_type_str += str(atype.itemsize * 8 // 2)
    elif atype.kind == "f":
        data_type_str += "rf"
        data_type_str += str(atype.itemsize * 8)  # itemsize in bits
    elif atype.kind in ("u", "i"):
        data_type_str += "r" + atype.kind
        data_type_str += str(atype.itemsize * 8)  # itemsize in bits
    if atype.itemsize > 1:
        # only append endianness for types over 8 bits
        data_type_str += get_endian_str(ray)
    return data_type_str

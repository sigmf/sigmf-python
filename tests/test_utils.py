# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Utilities"""

from datetime import datetime

import pytest

from sigmf import utils


@pytest.mark.parametrize(
    "ts, expected",
    [
        ("1955-07-04T05:15:00Z", datetime(year=1955, month=7, day=4, hour=5, minute=15, second=00, microsecond=0)),
        ("2956-08-05T06:15:12Z", datetime(year=2956, month=8, day=5, hour=6, minute=15, second=12, microsecond=0)),
        (
            "3957-09-06T07:15:12.345Z",
            datetime(year=3957, month=9, day=6, hour=7, minute=15, second=12, microsecond=345000),
        ),
        (
            "4958-10-07T08:15:12.0345Z",
            datetime(year=4958, month=10, day=7, hour=8, minute=15, second=12, microsecond=34500),
        ),
        (
            "5959-11-08T09:15:12.000000Z",
            datetime(year=5959, month=11, day=8, hour=9, minute=15, second=12, microsecond=0),
        ),
        (
            "6960-12-09T10:15:12.123456789123Z",
            datetime(year=6960, month=12, day=9, hour=10, minute=15, second=12, microsecond=123456),
        ),
    ],
)
def test_parse_simple_iso8601(ts, expected):
    dt = utils.parse_iso8601_datetime(ts)
    assert dt == expected

# Copyright: Multiple Authors
#
# This file is part of SigMF. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Provides pytest fixtures for other tests."""

import tempfile

import pytest

from sigmf import __specification__
from sigmf.sigmffile import SigMFFile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


@pytest.fixture
def test_data_file():
    """when called, yields temporary file"""
    with tempfile.NamedTemporaryFile() as temp:
        TEST_FLOAT32_DATA.tofile(temp.name)
        yield temp


@pytest.fixture
def test_sigmffile(test_data_file):
    """If pytest uses this signature, will return valid SigMF file."""
    sigf = SigMFFile()
    sigf.set_global_field("core:datatype", "rf32_le")
    sigf.set_global_field("core:version", __specification__)
    sigf.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA))
    sigf.add_capture(start_index=0)
    sigf.set_data_file(test_data_file.name)
    assert sigf._metadata == TEST_METADATA
    return sigf

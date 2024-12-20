# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Provides pytest fixtures for other tests."""

import tempfile

import pytest

from sigmf import __specification__
from sigmf.archive import SIGMF_DATASET_EXT
from sigmf.sigmffile import SigMFFile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


@pytest.fixture
def test_data_file():
    """when called, yields temporary dataset"""
    with tempfile.NamedTemporaryFile(suffix=f".{SIGMF_DATASET_EXT}") as temp:
        TEST_FLOAT32_DATA.tofile(temp.name)
        yield temp


@pytest.fixture
def test_sigmffile(test_data_file):
    """If pytest uses this signature, will return valid SigMF file."""
    meta = SigMFFile()
    meta.set_global_field("core:datatype", "rf32_le")
    meta.set_global_field("core:version", __specification__)
    meta.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA))
    meta.add_capture(start_index=0)
    meta.set_data_file(test_data_file.name)
    assert meta._metadata == TEST_METADATA
    return meta

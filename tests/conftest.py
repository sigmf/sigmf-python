# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Provides pytest fixtures for other tests."""

import os
import tempfile
from pathlib import Path

import pytest

from sigmf import DATATYPE_KEY, VERSION_KEY, __specification__
from sigmf.sigmffile import SigMFFile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


def get_nonsigmf_path() -> Path:
    """Get path to example_nonsigmf_recordings repo or skip test"""
    nonsigmf_env = "EXAMPLE_NONSIGMF_RECORDINGS_PATH"
    recordings_path = Path(os.getenv(nonsigmf_env, "nopath"))
    if not recordings_path.is_dir():
        pytest.skip(
            f"Set {nonsigmf_env} environment variable to path non-SigMF recordings repository to run test."
            f" Available at https://github.com/sigmf/example_nonsigmf_recordings"
        )
    return recordings_path


def validate_ncd(meta: SigMFFile, target_path: Path):
    """Validate that a SigMF object is a properly structured non-conforming dataset (NCD)."""
    assert str(meta.data_file) == str(target_path), "Auto-detected NCD should point to original file"
    assert isinstance(meta, SigMFFile)

    global_info = meta.get_global_info()
    capture_info = meta.get_captures()

    # validate NCD SigMF spec compliance
    assert len(capture_info) > 0, "Should have at least one capture"
    assert "core:header_bytes" in capture_info[0]
    if target_path.suffix != ".iq":
        # skip for Signal Hound
        assert capture_info[0]["core:header_bytes"] > 0, "Should have non-zero core:header_bytes field"
    assert "core:trailing_bytes" in global_info, "Should have core:trailing_bytes field."
    assert "core:dataset" in global_info, "Should have core:dataset field."
    assert "core:metadata_only" not in global_info, "Should NOT have core:metadata_only field."


@pytest.fixture
def test_data_file():
    """when called, yields temporary dataset"""
    with tempfile.NamedTemporaryFile(suffix=".sigmf-data") as temp:
        TEST_FLOAT32_DATA.tofile(temp.name)
        yield temp


@pytest.fixture
def test_sigmffile(test_data_file):
    """If pytest uses this signature, will return valid SigMF file."""
    meta = SigMFFile()
    meta.set_global_field(DATATYPE_KEY, "rf32_le")
    meta.set_global_field(VERSION_KEY, __specification__)
    meta.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA))
    meta.add_capture(start_index=0)
    meta.set_data_file(test_data_file.name)
    assert meta._metadata == TEST_METADATA
    return meta

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Provides pytest fixtures for other tests."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sigmf
from sigmf import DATATYPE_KEY, VERSION_KEY, __specification__
from sigmf.sigmffile import SigMFFile

TEST_FLOAT32_DATA = np.arange(16, dtype=np.float32)
TEST_METADATA = {
    SigMFFile.ANNOTATION_KEY: [{sigmf.SAMPLE_COUNT_KEY: 16, sigmf.SAMPLE_START_KEY: 0}],
    SigMFFile.CAPTURE_KEY: [{sigmf.SAMPLE_START_KEY: 0}],
    SigMFFile.GLOBAL_KEY: {
        sigmf.DATATYPE_KEY: "rf32_le",
        sigmf.SHA512_KEY: "f4984219b318894fa7144519185d1ae81ea721c6113243a52b51e444512a39d74cf41a4cec3c5d000bd7277cc71232c04d7a946717497e18619bdbe94bfeadd6",
        sigmf.NUM_CHANNELS_KEY: 1,
        sigmf.OFFSET_KEY: 0,
        sigmf.VERSION_KEY: __specification__,
    },
}


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

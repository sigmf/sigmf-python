# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Shared test data for tests."""

import os
import unittest
from pathlib import Path

import numpy as np

from sigmf import SigMFFile, __specification__, __version__


def get_nonsigmf_path(test: unittest.TestCase) -> Path:
    """Get path to example_nonsigmf_recordings repo or skip test"""
    nonsigmf_env = "EXAMPLE_NONSIGMF_RECORDINGS_PATH"
    recordings_path = Path(os.getenv(nonsigmf_env, "nopath"))
    if not recordings_path.is_dir():
        test.skipTest(
            f"Set {nonsigmf_env} environment variable to path non-SigMF recordings repository to run test."
            f" Available at https://github.com/sigmf/example_nonsigmf_recordings"
        )
    return recordings_path


def validate_ncd(test: unittest.TestCase, meta: SigMFFile, target_path: Path):
    """Validate that a SigMF object is a properly structured non-conforming dataset (NCD)."""
    test.assertEqual(str(meta.data_file), str(target_path), "Auto-detected NCD should point to original file")
    test.assertIsInstance(meta, SigMFFile)

    global_info = meta.get_global_info()
    capture_info = meta.get_captures()

    # validate NCD SigMF spec compliance
    test.assertGreater(len(capture_info), 0, "Should have at least one capture")
    test.assertIn("core:header_bytes", capture_info[0])
    if target_path.suffix != ".iq":
        # skip for Signal Hound
        test.assertGreater(capture_info[0]["core:header_bytes"], 0, "Should have non-zero core:header_bytes field")
    test.assertIn("core:trailing_bytes", global_info, "Should have core:trailing_bytes field.")
    test.assertIn("core:dataset", global_info, "Should have core:dataset field.")
    test.assertNotIn("core:metadata_only", global_info, "Should NOT have core:metadata_only field.")


TEST_FLOAT32_DATA = np.arange(16, dtype=np.float32)
TEST_METADATA = {
    SigMFFile.ANNOTATION_KEY: [{SigMFFile.SAMPLE_COUNT_KEY: 16, SigMFFile.SAMPLE_START_KEY: 0}],
    SigMFFile.CAPTURE_KEY: [{SigMFFile.SAMPLE_START_KEY: 0}],
    SigMFFile.GLOBAL_KEY: {
        SigMFFile.DATATYPE_KEY: "rf32_le",
        SigMFFile.SHA512_KEY: "f4984219b318894fa7144519185d1ae81ea721c6113243a52b51e444512a39d74cf41a4cec3c5d000bd7277cc71232c04d7a946717497e18619bdbe94bfeadd6",
        SigMFFile.NUM_CHANNELS_KEY: 1,
        SigMFFile.OFFSET_KEY: 0,
        SigMFFile.VERSION_KEY: __specification__,
    },
}

# Data0 is a test of a compliant two capture recording
TEST_U8_DATA0 = list(range(256))
TEST_U8_META0 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 0},
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 0},
    ],  # very strange..but technically legal?
    SigMFFile.GLOBAL_KEY: {SigMFFile.DATATYPE_KEY: "ru8", SigMFFile.TRAILING_BYTES_KEY: 0},
}
# Data1 is a test of a two capture recording with header_bytes and trailing_bytes set
TEST_U8_DATA1 = [0xFE] * 32 + list(range(192)) + [0xFF] * 32
TEST_U8_META1 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 32},
        {SigMFFile.SAMPLE_START_KEY: 128},
    ],
    SigMFFile.GLOBAL_KEY: {SigMFFile.DATATYPE_KEY: "ru8", SigMFFile.TRAILING_BYTES_KEY: 32},
}
# Data2 is a test of a two capture recording with multiple header_bytes set
TEST_U8_DATA2 = [0xFE] * 32 + list(range(128)) + [0xFE] * 16 + list(range(128, 192)) + [0xFF] * 16
TEST_U8_META2 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 32},
        {SigMFFile.SAMPLE_START_KEY: 128, SigMFFile.HEADER_BYTES_KEY: 16},
    ],
    SigMFFile.GLOBAL_KEY: {SigMFFile.DATATYPE_KEY: "ru8", SigMFFile.TRAILING_BYTES_KEY: 16},
}
# Data3 is a test of a three capture recording with multiple header_bytes set
TEST_U8_DATA3 = [0xFE] * 32 + list(range(128)) + [0xFE] * 32 + list(range(128, 192))
TEST_U8_META3 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 32},
        {SigMFFile.SAMPLE_START_KEY: 32},
        {SigMFFile.SAMPLE_START_KEY: 128, SigMFFile.HEADER_BYTES_KEY: 32},
    ],
    SigMFFile.GLOBAL_KEY: {SigMFFile.DATATYPE_KEY: "ru8"},
}
# Data4 is a two channel version of Data0
TEST_U8_DATA4 = [0xFE] * 32 + [y for y in list(range(96)) for i in [0, 1]] + [0xFF] * 32
TEST_U8_META4 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {SigMFFile.SAMPLE_START_KEY: 0, SigMFFile.HEADER_BYTES_KEY: 32},
        {SigMFFile.SAMPLE_START_KEY: 64},
    ],
    SigMFFile.GLOBAL_KEY: {
        SigMFFile.DATATYPE_KEY: "ru8",
        SigMFFile.TRAILING_BYTES_KEY: 32,
        SigMFFile.NUM_CHANNELS_KEY: 2,
    },
}

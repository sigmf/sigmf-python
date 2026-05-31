# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Shared test data for tests."""

import numpy as np

import sigmf
from sigmf import SigMFFile, __specification__

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

# Data0 is a test of a compliant two capture recording
TEST_U8_DATA0 = list(range(256))
TEST_U8_META0 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 0},
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 0},
    ],  # very strange..but technically legal?
    SigMFFile.GLOBAL_KEY: {sigmf.DATATYPE_KEY: "ru8", sigmf.TRAILING_BYTES_KEY: 0},
}
# Data1 is a test of a two capture recording with header_bytes and trailing_bytes set
TEST_U8_DATA1 = [0xFE] * 32 + list(range(192)) + [0xFF] * 32
TEST_U8_META1 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 32},
        {sigmf.SAMPLE_START_KEY: 128},
    ],
    SigMFFile.GLOBAL_KEY: {sigmf.DATATYPE_KEY: "ru8", sigmf.TRAILING_BYTES_KEY: 32},
}
# Data2 is a test of a two capture recording with multiple header_bytes set
TEST_U8_DATA2 = [0xFE] * 32 + list(range(128)) + [0xFE] * 16 + list(range(128, 192)) + [0xFF] * 16
TEST_U8_META2 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 32},
        {sigmf.SAMPLE_START_KEY: 128, sigmf.HEADER_BYTES_KEY: 16},
    ],
    SigMFFile.GLOBAL_KEY: {sigmf.DATATYPE_KEY: "ru8", sigmf.TRAILING_BYTES_KEY: 16},
}
# Data3 is a test of a three capture recording with multiple header_bytes set
TEST_U8_DATA3 = [0xFE] * 32 + list(range(128)) + [0xFE] * 32 + list(range(128, 192))
TEST_U8_META3 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 32},
        {sigmf.SAMPLE_START_KEY: 32},
        {sigmf.SAMPLE_START_KEY: 128, sigmf.HEADER_BYTES_KEY: 32},
    ],
    SigMFFile.GLOBAL_KEY: {sigmf.DATATYPE_KEY: "ru8"},
}
# Data4 is a two channel version of Data0
TEST_U8_DATA4 = [0xFE] * 32 + [y for y in list(range(96)) for i in [0, 1]] + [0xFF] * 32
TEST_U8_META4 = {
    SigMFFile.ANNOTATION_KEY: [],
    SigMFFile.CAPTURE_KEY: [
        {sigmf.SAMPLE_START_KEY: 0, sigmf.HEADER_BYTES_KEY: 32},
        {sigmf.SAMPLE_START_KEY: 64},
    ],
    SigMFFile.GLOBAL_KEY: {
        sigmf.DATATYPE_KEY: "ru8",
        sigmf.TRAILING_BYTES_KEY: 32,
        sigmf.NUM_CHANNELS_KEY: 2,
    },
}

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for edge-case datasets: header/trailing bytes, multiple captures, multiple channels."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sigmf
from sigmf import SigMFFile, utils

from .conftest import TEST_FLOAT32_DATA, TEST_METADATA

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


class TestCaptures(unittest.TestCase):
    """ensure capture access tools work properly"""

    def setUp(self) -> None:
        """ensure tests have a valid SigMF object to work with"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_path_data = self.temp_dir / "trash.sigmf-data"
        self.temp_path_meta = self.temp_dir / "trash.sigmf-meta"

    def tearDown(self) -> None:
        """remove temporary dir"""
        shutil.rmtree(self.temp_dir)

    def prepare(self, data: list, meta: dict, dtype: type, autoscale: bool = True) -> SigMFFile:
        """write some data and metadata to temporary paths"""
        np.array(data, dtype=dtype).tofile(self.temp_path_data)
        with open(self.temp_path_meta, "w") as handle:
            json.dump(meta, handle)
        meta = sigmf.fromfile(self.temp_path_meta, skip_checksum=True, autoscale=autoscale)
        return meta

    def test_compliant_two_capture_recording(self) -> None:
        """compliant two-capture recording"""
        meta = self.prepare(TEST_U8_DATA0, TEST_U8_META0, np.uint8, autoscale=False)
        self.assertEqual(256, meta._count_samples())
        self.assertTrue(meta._is_conforming_dataset())
        self.assertEqual((0, 0), meta.get_capture_byte_boundaries(0))
        self.assertEqual((0, 256), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(TEST_U8_DATA0, meta.read_samples()))
        self.assertTrue(np.array_equal(np.array([]), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(TEST_U8_DATA0, meta.read_samples_in_capture(1)))

    def test_two_capture_with_header_trailing_bytes(self) -> None:
        """two capture recording with header_bytes and trailing_bytes set"""
        meta = self.prepare(TEST_U8_DATA1, TEST_U8_META1, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertEqual((32, 160), meta.get_capture_byte_boundaries(0))
        self.assertEqual((160, 224), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(128), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(1)))

    def test_two_capture_with_multiple_header_bytes(self) -> None:
        """two capture recording with multiple header_bytes set"""
        meta = self.prepare(TEST_U8_DATA2, TEST_U8_META2, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertEqual((32, 160), meta.get_capture_byte_boundaries(0))
        self.assertEqual((176, 240), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(128), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(1)))

    def test_three_capture_with_multiple_header_bytes(self) -> None:
        """three capture recording with multiple header_bytes set"""
        meta = self.prepare(TEST_U8_DATA3, TEST_U8_META3, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertEqual((32, 64), meta.get_capture_byte_boundaries(0))
        self.assertEqual((64, 160), meta.get_capture_byte_boundaries(1))
        self.assertEqual((192, 256), meta.get_capture_byte_boundaries(2))
        self.assertTrue(np.array_equal(np.arange(32), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(32, 128), meta.read_samples_in_capture(1)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(2)))

    def test_two_channel_capture_recording(self) -> None:
        """two channel version of compliant capture recording"""
        meta = self.prepare(TEST_U8_DATA4, TEST_U8_META4, np.uint8, autoscale=False)
        self.assertEqual(96, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertEqual((32, 160), meta.get_capture_byte_boundaries(0))
        self.assertEqual((160, 224), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(64).repeat(2).reshape(-1, 2), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(64, 96).repeat(2).reshape(-1, 2), meta.read_samples_in_capture(1)))

    def test_slice_real_uint8(self) -> None:
        """slice real uint8"""
        meta = self.prepare(TEST_U8_DATA0, TEST_U8_META0, np.uint8, autoscale=False)
        self.assertTrue(np.array_equal(meta[:], TEST_U8_DATA0))
        self.assertTrue(np.array_equal(meta[6], TEST_U8_DATA0[6]))
        self.assertTrue(np.array_equal(meta[1:-1], TEST_U8_DATA0[1:-1]))

    def test_slice_real_float32(self) -> None:
        """slice real float32"""
        meta = self.prepare(TEST_FLOAT32_DATA, TEST_METADATA, np.float32)
        self.assertTrue(np.array_equal(meta[:], TEST_FLOAT32_DATA))
        self.assertTrue(np.array_equal(meta[9], TEST_FLOAT32_DATA[9]))

    def test_slice_multiple_channels(self) -> None:
        """slice multiple channels"""

        meta = self.prepare(TEST_U8_DATA4, TEST_U8_META4, np.uint8, autoscale=False)
        channelized = np.array(TEST_U8_DATA4).reshape((-1, 2))
        # the map is bounded to the sample data, so trailing bytes are not exposed
        trailing_bytes = TEST_U8_META4[SigMFFile.GLOBAL_KEY][sigmf.TRAILING_BYTES_KEY]
        self.assertTrue(np.array_equal(meta[:][:], channelized[: -trailing_bytes // 2]))
        self.assertTrue(np.array_equal(meta[10:20, 0], meta.read_samples()[10:20, 0]))
        self.assertTrue(np.array_equal(meta[0], channelized[0]))
        self.assertTrue(np.array_equal(meta[1, :], channelized[1]))

    def test_capture_byte_boundaries(self) -> None:
        """capture byte boundaries from pairs & archives"""
        # get a meta pair and archive
        meta = self.prepare(TEST_U8_DATA3, TEST_U8_META3, np.uint8)
        arc_path = self.temp_dir / "arc.sigmf"
        meta.tofile(arc_path)
        arc = sigmf.fromfile(arc_path)
        for bdx in range(3):
            self.assertEqual(meta.get_capture_byte_boundaries(bdx), arc.get_capture_byte_boundaries(bdx))
            self.assertTrue(np.array_equal(meta.read_samples_in_capture(bdx), arc.read_samples_in_capture(bdx)))

    def test_add_capture(self):
        """test basic capture addition"""
        meta = SigMFFile()
        meta.add_capture(start_index=0, metadata={})

    def test_add_capture_metadata_merge(self):
        """test that adding capture with existing start_index properly merges metadata"""
        meta = SigMFFile()

        # add initial capture with some metadata
        initial_meta = {"core:frequency": 915e6, "core:sample_rate": 1e6}
        meta.add_capture(start_index=0, metadata=initial_meta)

        # add capture with same start_index but additional metadata
        additional_meta = {"core:datetime": "2026-03-17T10:00:00Z", "custom:gain": 30}
        meta.add_capture(start_index=0, metadata=additional_meta)

        # verify metadata was merged properly
        captures = meta.get_captures()
        self.assertEqual(len(captures), 1, "should have exactly one capture")

        merged_capture = captures[0]
        # original metadata should be preserved
        self.assertEqual(merged_capture["core:frequency"], 915e6)
        self.assertEqual(merged_capture["core:sample_rate"], 1e6)
        # new metadata should be added
        self.assertEqual(merged_capture["core:datetime"], "2026-03-17T10:00:00Z")
        self.assertEqual(merged_capture["custom:gain"], 30)

    def test_add_multiple_captures_and_annotations(self):
        """test adding multiple captures with annotations"""
        meta = SigMFFile()
        for idx in range(3):
            simulate_capture(meta, idx, 1024)


def simulate_capture(sigmf_md, n, capture_len):
    start_index = capture_len * n

    capture_md = {"core:datetime": utils.get_sigmf_iso8601_datetime_now()}

    sigmf_md.add_capture(start_index=start_index, metadata=capture_md)

    annotation_md = {
        "core:latitude": 40.0 + 0.0001 * n,
        "core:longitude": -105.0 + 0.0001 * n,
    }

    sigmf_md.add_annotation(start_index=start_index, length=capture_len, metadata=annotation_md)

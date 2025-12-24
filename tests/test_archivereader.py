# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for SigMFArchiveReader"""

import unittest
from tempfile import NamedTemporaryFile

import numpy as np

import sigmf
from sigmf import SigMFArchiveReader, SigMFFile, __specification__


class TestArchiveReader(unittest.TestCase):
    def setUp(self):
        # in order to check shapes we need some positive number of samples to work with
        # number of samples should be lowest common factor of num_channels
        self.raw_count = 16
        self.lut = {
            "i8": np.int8,
            "u8": np.uint8,
            "i16": np.int16,
            "u16": np.uint16,
            "u32": np.uint32,
            "i32": np.int32,
            "f32": np.float32,
            "f64": np.float64,
        }

    def test_access_data_without_untar(self):
        """iterate through datatypes and verify IO is correct"""
        temp_data = NamedTemporaryFile()
        temp_archive = NamedTemporaryFile(suffix=".sigmf")

        for key, dtype in self.lut.items():
            # for each type of storage
            temp_samples = np.arange(self.raw_count, dtype=dtype)
            temp_samples.tofile(temp_data.name)
            for num_channels in [1, 4, 8]:
                # for single or 8 channel
                for complex_prefix in ["r", "c"]:
                    # for real or complex
                    target_count = self.raw_count
                    temp_meta = SigMFFile(
                        data_file=temp_data.name,
                        global_info={
                            SigMFFile.DATATYPE_KEY: f"{complex_prefix}{key}_le",
                            SigMFFile.NUM_CHANNELS_KEY: num_channels,
                        },
                    )
                    temp_meta.tofile(temp_archive.name, toarchive=True)

                    readback = SigMFArchiveReader(temp_archive.name)
                    readback_samples = readback[:]

                    if complex_prefix == "c":
                        # complex data will be half as long
                        target_count //= 2
                        self.assertTrue(np.iscomplexobj(readback_samples))
                    if num_channels != 1:
                        # check expected # of channels
                        self.assertEqual(
                            readback_samples.ndim,
                            2,
                            "Mismatch in shape of readback samples.",
                        )
                    target_count //= num_channels

                    self.assertEqual(
                        target_count,
                        temp_meta._count_samples(),
                        "Mismatch in expected metadata length.",
                    )
                    self.assertEqual(
                        target_count,
                        len(readback),
                        "Mismatch in expected readback length",
                    )


def test_archiveread_data_file_unchanged(test_sigmffile):
    with NamedTemporaryFile(suffix=".sigmf") as temp_file:
        input_samples = test_sigmffile.read_samples()
        test_sigmffile.archive(temp_file.name)
        arc = sigmf.fromfile(temp_file.name)
        output_samples = arc.read_samples()

        assert np.array_equal(input_samples, output_samples)

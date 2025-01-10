# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for SigMFFile Object"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sigmf import error, sigmffile, utils
from sigmf.sigmffile import SigMFFile

from .testdata import *


class TestClassMethods(unittest.TestCase):
    def setUp(self):
        """ensure tests have a valid SigMF object to work with"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_path_data = self.temp_dir / "trash.sigmf-data"
        self.temp_path_meta = self.temp_dir / "trash.sigmf-meta"
        TEST_FLOAT32_DATA.tofile(self.temp_path_data)
        self.sigmf_object = SigMFFile(TEST_METADATA, data_file=self.temp_path_data)
        self.sigmf_object.tofile(self.temp_path_meta)

    def tearDown(self):
        """remove temporary dir"""
        shutil.rmtree(self.temp_dir)

    def test_pathlib_handle(self):
        """ensure file can be a string or a pathlib object"""
        self.assertTrue(self.temp_path_data.exists())
        obj_str = sigmffile.fromfile(str(self.temp_path_data))
        obj_str.validate()
        obj_pth = sigmffile.fromfile(self.temp_path_data)
        obj_pth.validate()

    def test_iterator_basic(self):
        """make sure default batch_size works"""
        count = 0
        for _ in self.sigmf_object:
            count += 1
        self.assertEqual(count, len(self.sigmf_object))

    def test_checksum(self):
        """Ensure checksum fails when incorrect or empty string."""
        for new_checksum in ("", "a", 0):
            bad_checksum_metadata = copy.deepcopy(TEST_METADATA)
            bad_checksum_metadata[SigMFFile.GLOBAL_KEY][SigMFFile.HASH_KEY] = new_checksum
            with self.assertRaises(error.SigMFFileError):
                _ = SigMFFile(bad_checksum_metadata, self.temp_path_data)

    def test_equality(self):
        """Ensure __eq__ working as expected"""
        other = SigMFFile(copy.deepcopy(TEST_METADATA))
        self.assertEqual(self.sigmf_object, other)
        # different after changing any part of metadata
        other.add_annotation(start_index=0, metadata={"a": 0})
        self.assertNotEqual(self.sigmf_object, other)


class TestAnnotationHandling(unittest.TestCase):
    def test_get_annotations_with_index(self):
        """Test that only annotations containing index are returned from get_annotations()"""
        smf = SigMFFile(copy.deepcopy(TEST_METADATA))
        smf.add_annotation(start_index=1)
        smf.add_annotation(start_index=4, length=4)
        annotations_idx10 = smf.get_annotations(index=10)
        self.assertListEqual(
            annotations_idx10,
            [
                {SigMFFile.START_INDEX_KEY: 0, SigMFFile.LENGTH_INDEX_KEY: 16},
                {SigMFFile.START_INDEX_KEY: 1},
            ],
        )

    def test__count_samples_from_annotation(self):
        """Make sure sample count from annotations use correct end index"""
        smf = SigMFFile(copy.deepcopy(TEST_METADATA))
        smf.add_annotation(start_index=0, length=32)
        smf.add_annotation(start_index=4, length=4)
        sample_count = smf._count_samples()
        self.assertEqual(sample_count, 32)

    def test_set_data_file_without_annotations(self):
        """
        Make sure setting data_file with no annotations registered does not
        raise any errors
        """
        smf = SigMFFile(copy.deepcopy(TEST_METADATA))
        smf._metadata[SigMFFile.ANNOTATION_KEY].clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path_data = os.path.join(tmpdir, "datafile")
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            smf.set_data_file(temp_path_data)
            samples = smf.read_samples()
            self.assertTrue(len(samples) == 16)

    def test_set_data_file_with_annotations(self):
        """
        Make sure setting data_file with annotations registered use sample
        count from data_file and issue a warning if annotations have end
        indices bigger than file end index
        """
        smf = SigMFFile(copy.deepcopy(TEST_METADATA))
        smf.add_annotation(start_index=0, length=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path_data = os.path.join(tmpdir, "datafile")
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            with self.assertWarns(Warning):
                # Issues warning since file ends before the final annotatio
                smf.set_data_file(temp_path_data)
                samples = smf.read_samples()
                self.assertTrue(len(samples) == 16)


def simulate_capture(sigmf_md, n, capture_len):
    start_index = capture_len * n

    capture_md = {"core:datetime": utils.get_sigmf_iso8601_datetime_now()}

    sigmf_md.add_capture(start_index=start_index, metadata=capture_md)

    annotation_md = {
        "core:latitude": 40.0 + 0.0001 * n,
        "core:longitude": -105.0 + 0.0001 * n,
    }

    sigmf_md.add_annotation(start_index=start_index, length=capture_len, metadata=annotation_md)


def test_default_constructor():
    SigMFFile()


def test_set_non_required_global_field():
    sigf = SigMFFile()
    sigf.set_global_field("this_is:not_in_the_schema", None)


def test_add_capture():
    sigf = SigMFFile()
    sigf.add_capture(start_index=0, metadata={})


def test_add_annotation():
    sigf = SigMFFile()
    sigf.add_capture(start_index=0)
    meta = {"latitude": 40.0, "longitude": -105.0}
    sigf.add_annotation(start_index=0, length=128, metadata=meta)


def test_fromarchive(test_sigmffile):
    print("test_sigmffile is:\n", test_sigmffile)
    tf = tempfile.mkstemp()[1]
    td = tempfile.mkdtemp()
    archive_path = test_sigmffile.archive(name=tf)
    result = sigmffile.fromarchive(archive_path=archive_path, dir=td)
    assert result._metadata == test_sigmffile._metadata == TEST_METADATA
    os.remove(tf)
    shutil.rmtree(td)


def test_add_multiple_captures_and_annotations():
    sigf = SigMFFile()
    for idx in range(3):
        simulate_capture(sigf, idx, 1024)


class TestMultichannel(unittest.TestCase):
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

    def test_multichannel_types(self):
        """check that real & complex for all types is reading multiple channels correctly"""
        _, temp_path = tempfile.mkstemp()
        for key, dtype in self.lut.items():
            # for each type of storage
            np.arange(self.raw_count, dtype=dtype).tofile(temp_path)
            for num_channels in [1, 4, 8]:
                # for single or 8 channel
                for complex_prefix in ["r", "c"]:
                    # for real or complex
                    check_count = self.raw_count
                    temp_signal = SigMFFile(
                        data_file=temp_path,
                        global_info={
                            SigMFFile.DATATYPE_KEY: f"{complex_prefix}{key}_le",
                            SigMFFile.NUM_CHANNELS_KEY: num_channels,
                        },
                    )
                    temp_samples = temp_signal.read_samples()

                    if complex_prefix == "c":
                        # complex data will be half as long
                        check_count //= 2
                        self.assertTrue(np.all(np.iscomplex(temp_samples)))
                    if num_channels != 1:
                        self.assertEqual(temp_samples.ndim, 2)
                    check_count //= num_channels

                    self.assertEqual(check_count, temp_signal._count_samples())

    def test_multichannel_seek(self):
        """assure that seeking is working correctly with multichannel files"""
        _, temp_path = tempfile.mkstemp()
        # write some dummy data and read back
        np.arange(18, dtype=np.uint16).tofile(temp_path)
        temp_signal = SigMFFile(
            data_file=temp_path,
            global_info={
                SigMFFile.DATATYPE_KEY: "cu16_le",
                SigMFFile.NUM_CHANNELS_KEY: 3,
            },
        )
        # read after the first sample
        temp_samples = temp_signal.read_samples(start_index=1, autoscale=False)
        # assure samples are in the order we expect
        self.assertTrue(np.all(temp_samples[:, 0] == np.array([6 + 7j, 12 + 13j])))


def test_key_validity():
    """assure the keys in test metadata are valid"""
    for top_key, top_val in TEST_METADATA.items():
        if type(top_val) is dict:
            for core_key in top_val.keys():
                assert core_key in vars(SigMFFile)[f"VALID_{top_key.upper()}_KEYS"]
        elif type(top_val) is list:
            # annotations are in a list
            for annot in top_val:
                for core_key in annot.keys():
                    assert core_key in SigMFFile.VALID_ANNOTATION_KEYS
        else:
            raise ValueError("expected list or dict")


def test_ordered_metadata():
    """check to make sure the metadata is sorted as expected"""
    sigf = SigMFFile()
    top_sort_order = ["global", "captures", "annotations"]
    for kdx, key in enumerate(sigf.ordered_metadata()):
        assert kdx == top_sort_order.index(key)


def test_captures_checking():
    """
    these tests make sure the various captures access tools work properly
    """
    np.array(TEST_U8_DATA0, dtype=np.uint8).tofile("/tmp/d0.sigmf-data")
    with open("/tmp/d0.sigmf-meta", "w") as f0:
        json.dump(TEST_U8_META0, f0)
    np.array(TEST_U8_DATA1, dtype=np.uint8).tofile("/tmp/d1.sigmf-data")
    with open("/tmp/d1.sigmf-meta", "w") as f1:
        json.dump(TEST_U8_META1, f1)
    np.array(TEST_U8_DATA2, dtype=np.uint8).tofile("/tmp/d2.sigmf-data")
    with open("/tmp/d2.sigmf-meta", "w") as f2:
        json.dump(TEST_U8_META2, f2)
    np.array(TEST_U8_DATA3, dtype=np.uint8).tofile("/tmp/d3.sigmf-data")
    with open("/tmp/d3.sigmf-meta", "w") as f3:
        json.dump(TEST_U8_META3, f3)
    np.array(TEST_U8_DATA4, dtype=np.uint8).tofile("/tmp/d4.sigmf-data")
    with open("/tmp/d4.sigmf-meta", "w") as f4:
        json.dump(TEST_U8_META4, f4)

    sigmf0 = sigmffile.fromfile("/tmp/d0.sigmf-meta", skip_checksum=True)
    sigmf1 = sigmffile.fromfile("/tmp/d1.sigmf-meta", skip_checksum=True)
    sigmf2 = sigmffile.fromfile("/tmp/d2.sigmf-meta", skip_checksum=True)
    sigmf3 = sigmffile.fromfile("/tmp/d3.sigmf-meta", skip_checksum=True)
    sigmf4 = sigmffile.fromfile("/tmp/d4.sigmf-meta", skip_checksum=True)

    assert sigmf0._count_samples() == 256
    assert sigmf0._is_conforming_dataset()
    assert (0, 0) == sigmf0.get_capture_byte_boundarys(0)
    assert (0, 256) == sigmf0.get_capture_byte_boundarys(1)
    assert np.array_equal(TEST_U8_DATA0, sigmf0.read_samples(autoscale=False))
    assert np.array_equal(np.array([]), sigmf0.read_samples_in_capture(0))
    assert np.array_equal(TEST_U8_DATA0, sigmf0.read_samples_in_capture(1, autoscale=False))

    assert sigmf1._count_samples() == 192
    assert not sigmf1._is_conforming_dataset()
    assert (32, 160) == sigmf1.get_capture_byte_boundarys(0)
    assert (160, 224) == sigmf1.get_capture_byte_boundarys(1)
    assert np.array_equal(np.array(range(128)), sigmf1.read_samples_in_capture(0, autoscale=False))
    assert np.array_equal(np.array(range(128, 192)), sigmf1.read_samples_in_capture(1, autoscale=False))

    assert sigmf2._count_samples() == 192
    assert not sigmf2._is_conforming_dataset()
    assert (32, 160) == sigmf2.get_capture_byte_boundarys(0)
    assert (176, 240) == sigmf2.get_capture_byte_boundarys(1)
    assert np.array_equal(np.array(range(128)), sigmf2.read_samples_in_capture(0, autoscale=False))
    assert np.array_equal(np.array(range(128, 192)), sigmf2.read_samples_in_capture(1, autoscale=False))

    assert sigmf3._count_samples() == 192
    assert not sigmf3._is_conforming_dataset()
    assert (32, 64) == sigmf3.get_capture_byte_boundarys(0)
    assert (64, 160) == sigmf3.get_capture_byte_boundarys(1)
    assert (192, 256) == sigmf3.get_capture_byte_boundarys(2)
    assert np.array_equal(np.array(range(32)), sigmf3.read_samples_in_capture(0, autoscale=False))
    assert np.array_equal(np.array(range(32, 128)), sigmf3.read_samples_in_capture(1, autoscale=False))
    assert np.array_equal(np.array(range(128, 192)), sigmf3.read_samples_in_capture(2, autoscale=False))

    assert sigmf4._count_samples() == 96
    assert not sigmf4._is_conforming_dataset()
    assert (32, 160) == sigmf4.get_capture_byte_boundarys(0)
    assert (160, 224) == sigmf4.get_capture_byte_boundarys(1)
    assert np.array_equal(np.array(range(64)), sigmf4.read_samples_in_capture(0, autoscale=False)[:, 0])
    assert np.array_equal(np.array(range(64, 96)), sigmf4.read_samples_in_capture(1, autoscale=False)[:, 1])


def test_slicing():
    """Test __getitem___ builtin for sigmffile, make sure slicing and indexing works as expected."""
    _, temp_data0 = tempfile.mkstemp()
    np.array(TEST_U8_DATA0, dtype=np.uint8).tofile(temp_data0)
    sigmf0 = SigMFFile(metadata=TEST_U8_META0, data_file=temp_data0)
    assert np.array_equal(TEST_U8_DATA0, sigmf0[:])
    assert TEST_U8_DATA0[6] == sigmf0[6]

    # test float32
    _, temp_data1 = tempfile.mkstemp()
    np.array(TEST_FLOAT32_DATA, dtype=np.float32).tofile(temp_data1)
    sigmf1 = SigMFFile(metadata=TEST_METADATA, data_file=temp_data1)
    assert np.array_equal(TEST_FLOAT32_DATA, sigmf1[:])
    assert sigmf1[10] == TEST_FLOAT32_DATA[10]

    # test multiple channels
    _, temp_data2 = tempfile.mkstemp()
    np.array(TEST_U8_DATA4, dtype=np.uint8).tofile(temp_data2)
    sigmf2 = SigMFFile(TEST_U8_META4, data_file=temp_data2)
    channelized = np.array(TEST_U8_DATA4).reshape((128, 2))
    assert np.array_equal(channelized, sigmf2[:][:])
    assert np.array_equal(sigmf2[10:20, 91:112], sigmf2.read_samples(autoscale=False)[10:20, 91:112])
    assert np.array_equal(sigmf2[0], channelized[0])
    assert np.array_equal(sigmf2[1, :], channelized[1, :])

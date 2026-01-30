# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for SigMFFile Object"""

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sigmf
from sigmf import SigMFFile, error, utils

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
        obj_str = sigmf.fromfile(str(self.temp_path_data))
        obj_str.validate()
        obj_pth = sigmf.fromfile(self.temp_path_data)
        obj_pth.validate()

    def test_filenames_with_dots(self):
        """test that filenames with non-extension . characters are handled correctly"""
        filenames = ["a", "b.c", "d.e.f"]
        for filename in filenames:
            temp_path_data = self.temp_dir / f"{filename}.sigmf-data"
            temp_path_meta = self.temp_dir / f"{filename}.sigmf-meta"
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            self.sigmf_object = SigMFFile(TEST_METADATA, data_file=temp_path_data)
            self.sigmf_object.tofile(temp_path_meta)
            files = [str(temp_path_data), temp_path_data, str(temp_path_meta), temp_path_meta]
            for filename in files:
                obj = sigmf.fromfile(filename)
                obj.validate()

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
            temp_path_data = Path(tmpdir) / "datafile"
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
            temp_path_data = Path(tmpdir) / "datafile"
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            with self.assertWarns(Warning):
                # Issues warning since file ends before the final annotatio
                smf.set_data_file(temp_path_data)
                samples = smf.read_samples()
                self.assertTrue(len(samples) == 16)


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
        self.temp_file = tempfile.NamedTemporaryFile()
        self.temp_path = Path(self.temp_file.name)

    def tearDown(self):
        """clean-up temporary files"""
        self.temp_file.close()

    def test_multichannel_types(self):
        """check that real & complex for all types is reading multiple channels correctly"""
        for key, dtype in self.lut.items():
            # for each type of storage
            np.arange(self.raw_count, dtype=dtype).tofile(self.temp_path)
            for num_channels in [1, 4, 8]:
                # for single or 8 channel
                for complex_prefix in ["r", "c"]:
                    # for real or complex
                    check_count = self.raw_count
                    temp_signal = SigMFFile(
                        data_file=self.temp_path,
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
        """ensure that seeking is working correctly with multichannel files"""
        # write some dummy data and read back
        np.arange(18, dtype=np.uint16).tofile(self.temp_path)
        temp_signal = SigMFFile(
            data_file=self.temp_path,
            global_info={
                SigMFFile.DATATYPE_KEY: "cu16_le",
                SigMFFile.NUM_CHANNELS_KEY: 3,
            },
            autoscale=False,
        )
        # read after the first sample
        temp_samples = temp_signal.read_samples(start_index=1)
        # ensure samples are in the order we expect
        self.assertTrue(np.all(temp_samples[:, 0] == np.array([6 + 7j, 12 + 13j])))


def test_key_validity():
    """ensure the keys in test metadata are valid"""
    for top_key, top_val in TEST_METADATA.items():
        if isinstance(top_val, dict):
            for core_key in top_val.keys():
                assert core_key in vars(SigMFFile)[f"VALID_{top_key.upper()}_KEYS"]
        elif isinstance(top_val, list):
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

    def test_000(self) -> None:
        """compliant two-capture recording"""
        meta = self.prepare(TEST_U8_DATA0, TEST_U8_META0, np.uint8, autoscale=False)
        self.assertEqual(256, meta._count_samples())
        self.assertTrue(meta._is_conforming_dataset())
        self.assertTrue((0, 0), meta.get_capture_byte_boundaries(0))
        self.assertTrue((0, 256), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(TEST_U8_DATA0, meta.read_samples()))
        self.assertTrue(np.array_equal(np.array([]), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(TEST_U8_DATA0, meta.read_samples_in_capture(1)))

    def test_001(self) -> None:
        """two capture recording with header_bytes and trailing_bytes set"""
        meta = self.prepare(TEST_U8_DATA1, TEST_U8_META1, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertTrue((32, 160), meta.get_capture_byte_boundaries(0))
        self.assertTrue((160, 224), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(128), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(1)))

    def test_002(self) -> None:
        """two capture recording with multiple header_bytes set"""
        meta = self.prepare(TEST_U8_DATA2, TEST_U8_META2, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertTrue((32, 160), meta.get_capture_byte_boundaries(0))
        self.assertTrue((160, 224), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(128), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(1)))

    def test_003(self) -> None:
        """three capture recording with multiple header_bytes set"""
        meta = self.prepare(TEST_U8_DATA3, TEST_U8_META3, np.uint8, autoscale=False)
        self.assertEqual(192, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertTrue((32, 64), meta.get_capture_byte_boundaries(0))
        self.assertTrue((64, 160), meta.get_capture_byte_boundaries(1))
        self.assertTrue((160, 224), meta.get_capture_byte_boundaries(2))
        self.assertTrue(np.array_equal(np.arange(32), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(32, 128), meta.read_samples_in_capture(1)))
        self.assertTrue(np.array_equal(np.arange(128, 192), meta.read_samples_in_capture(2)))

    def test_004(self) -> None:
        """two channel version of 000"""
        meta = self.prepare(TEST_U8_DATA4, TEST_U8_META4, np.uint8, autoscale=False)
        self.assertEqual(96, meta._count_samples())
        self.assertFalse(meta._is_conforming_dataset())
        self.assertTrue((32, 96), meta.get_capture_byte_boundaries(0))
        self.assertTrue((96, 160), meta.get_capture_byte_boundaries(1))
        self.assertTrue(np.array_equal(np.arange(64).repeat(2).reshape(-1, 2), meta.read_samples_in_capture(0)))
        self.assertTrue(np.array_equal(np.arange(64, 96).repeat(2).reshape(-1, 2), meta.read_samples_in_capture(1)))

    def test_slicing_ru8(self) -> None:
        """slice real uint8"""
        meta = self.prepare(TEST_U8_DATA0, TEST_U8_META0, np.uint8, autoscale=False)
        self.assertTrue(np.array_equal(meta[:], TEST_U8_DATA0))
        self.assertTrue(np.array_equal(meta[6], TEST_U8_DATA0[6]))
        self.assertTrue(np.array_equal(meta[1:-1], TEST_U8_DATA0[1:-1]))

    def test_slicing_rf32(self) -> None:
        """slice real float32"""
        meta = self.prepare(TEST_FLOAT32_DATA, TEST_METADATA, np.float32)
        self.assertTrue(np.array_equal(meta[:], TEST_FLOAT32_DATA))
        self.assertTrue(np.array_equal(meta[9], TEST_FLOAT32_DATA[9]))

    def test_slicing_multiple_channels(self) -> None:
        """slice multiple channels"""

        meta = self.prepare(TEST_U8_DATA4, TEST_U8_META4, np.uint8, autoscale=False)
        channelized = np.array(TEST_U8_DATA4).reshape((-1, 2))
        self.assertTrue(np.array_equal(meta[:][:], channelized))
        self.assertTrue(np.array_equal(meta[10:20, 0], meta.read_samples()[10:20, 0]))
        self.assertTrue(np.array_equal(meta[0], channelized[0]))
        self.assertTrue(np.array_equal(meta[1, :], channelized[1]))

    def test_boundaries(self) -> None:
        """capture byte boundaries from pairs & archives"""
        # get a meta pair and archive
        meta = self.prepare(TEST_U8_DATA3, TEST_U8_META3, np.uint8)
        arc_path = self.temp_dir / "arc.sigmf"
        meta.tofile(arc_path, toarchive=True)
        arc = sigmf.fromfile(arc_path)
        for bdx in range(3):
            self.assertEqual(meta.get_capture_byte_boundaries(bdx), arc.get_capture_byte_boundaries(bdx))
            self.assertTrue(np.array_equal(meta.read_samples_in_capture(bdx), arc.read_samples_in_capture(bdx)))


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
    with tempfile.NamedTemporaryFile(suffix=".sigmf") as temp_file:
        archive_path = test_sigmffile.archive(name=temp_file.name, overwrite=True)
        result = sigmf.fromarchive(archive_path=archive_path)
        assert result._metadata == test_sigmffile._metadata == TEST_METADATA


def test_add_multiple_captures_and_annotations():
    sigf = SigMFFile()
    for idx in range(3):
        simulate_capture(sigf, idx, 1024)


class TestOverwrite(unittest.TestCase):
    """test file overwrite protection"""

    def setUp(self):
        """create temporary directory and test files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_path = self.temp_dir / "test.sigmf-data"
        self.test_meta_path = self.temp_dir / "test.sigmf-meta"
        self.test_archive_path = self.temp_dir / "test.sigmf"
        self.test_collection_path = self.temp_dir / "test.sigmf-collection"

        # write test data file
        TEST_FLOAT32_DATA.tofile(self.test_data_path)

        # create test sigmf object
        self.sigmf_obj = SigMFFile(TEST_METADATA, data_file=self.test_data_path)

        # create alternate test data for overwrite testing
        self.alt_data = np.arange(16, 32, dtype=np.float32)  # different data for checksum verification
        self.alt_data_path = self.temp_dir / "alt.sigmf-data"
        self.alt_data.tofile(self.alt_data_path)

    def tearDown(self):
        """clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_prevent_metadata_overwrite(self):
        """tofile raises exception when metadata file exists and overwrite=False"""
        # create existing metadata file
        self.sigmf_obj.tofile(self.test_meta_path)
        with self.assertRaises(error.SigMFFileError) as context:
            self.sigmf_obj.tofile(self.test_meta_path, overwrite=False)
        self.assertIn("already exists", str(context.exception))

    def test_metadata_overwrite_works(self):
        """tofile succeeds when metadata file exists and overwrite=True"""
        # create existing metadata file
        self.sigmf_obj.tofile(self.test_meta_path)
        self.assertTrue(self.test_meta_path.exists())
        original_content = self.test_meta_path.read_text()
        original_checksum = self.sigmf_obj.get_global_field("core:sha512")

        # create sigmf object with different data and metadata
        alt_sigmf = SigMFFile()
        alt_sigmf.set_global_field(SigMFFile.DATATYPE_KEY, "rf32_le")
        alt_sigmf.set_global_field("core:description", "overwritten file")
        alt_sigmf.set_data_file(self.alt_data_path)

        # should succeed with overwrite=True and content should change
        alt_sigmf.tofile(self.test_meta_path, overwrite=True)
        self.assertTrue(self.test_meta_path.exists())
        new_content = self.test_meta_path.read_text()
        new_checksum = alt_sigmf.get_global_field("core:sha512")

        self.assertNotEqual(original_content, new_content, "file content should change when overwritten")
        self.assertNotEqual(original_checksum, new_checksum, "SHA512 checksum should change when overwritten")

    def test_prevent_archive_overwrite(self):
        """tofile archive raises exception when archive exists and overwrite=False"""
        # create existing archive
        self.sigmf_obj.tofile(self.test_archive_path, toarchive=True)
        with self.assertRaises(error.SigMFFileError) as context:
            self.sigmf_obj.tofile(self.test_archive_path, toarchive=True, overwrite=False)
        self.assertIn("already exists", str(context.exception))

    def test_archive_overwrite_works(self):
        """tofile archive succeeds when archive exists and overwrite=True"""
        # create existing archive
        self.sigmf_obj.tofile(self.test_archive_path, toarchive=True)
        self.assertTrue(self.test_archive_path.exists())
        original_checksum = self.sigmf_obj.get_global_field("core:sha512")

        # create sigmf object with different data
        alt_sigmf = SigMFFile()
        alt_sigmf.set_global_field(SigMFFile.DATATYPE_KEY, "rf32_le")
        alt_sigmf.set_global_field("core:description", "overwritten archive")
        alt_sigmf.set_data_file(self.alt_data_path)

        # should succeed with overwrite=True and content should change
        alt_sigmf.tofile(self.test_archive_path, toarchive=True, overwrite=True)
        self.assertTrue(self.test_archive_path.exists())

        # verify by reading the archive content back
        readback_sigmf = sigmf.fromarchive(self.test_archive_path)
        new_checksum = readback_sigmf.get_global_field("core:sha512")

        self.assertEqual(readback_sigmf.get_global_field("core:description"), "overwritten archive")
        self.assertNotEqual(original_checksum, new_checksum, "SHA512 checksum should change when overwritten")

    def test_default_behavior(self):
        """overwrite defaults to False for safety"""
        # create existing files
        self.sigmf_obj.tofile(self.test_meta_path)
        self.sigmf_obj.tofile(self.test_archive_path, toarchive=True)

        # should raise exceptions with default overwrite=False
        with self.assertRaises(error.SigMFFileError):
            self.sigmf_obj.tofile(self.test_meta_path)

        with self.assertRaises(error.SigMFFileError):
            self.sigmf_obj.tofile(self.test_archive_path, toarchive=True)

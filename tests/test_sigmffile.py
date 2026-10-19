# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for SigMFFile Object"""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sigmf
from sigmf import SigMFFile, error

from .conftest import TEST_FLOAT32_DATA, TEST_METADATA


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
            bad_checksum_metadata[SigMFFile.GLOBAL_KEY][sigmf.SHA512_KEY] = new_checksum
            with self.assertRaises(error.SigMFFileError):
                _ = SigMFFile(bad_checksum_metadata, self.temp_path_data)

    def test_equality(self):
        """Ensure __eq__ working as expected"""
        other = SigMFFile(TEST_METADATA)
        self.assertEqual(self.sigmf_object, other)
        # different after changing any part of metadata
        other.add_annotation(start_index=0, metadata={"a": 0})
        self.assertNotEqual(self.sigmf_object, other)


class TestAnnotationHandling(unittest.TestCase):
    def test_get_annotations_with_index(self):
        """Test that only annotations containing index are returned from get_annotations()"""
        meta = SigMFFile(TEST_METADATA)
        meta.add_annotation(start_index=1)
        meta.add_annotation(start_index=4, length=4)
        annotations_idx10 = meta.get_annotations(index=10)
        self.assertListEqual(
            annotations_idx10,
            [
                {sigmf.SAMPLE_START_KEY: 0, sigmf.SAMPLE_COUNT_KEY: 16},
                {sigmf.SAMPLE_START_KEY: 1},
            ],
        )

    def test_sample_count_from_annotations(self):
        """Make sure sample count from annotations use correct end index"""
        meta = SigMFFile(TEST_METADATA)
        meta.add_annotation(start_index=0, length=32)
        meta.add_annotation(start_index=4, length=4)
        sample_count = meta._count_samples()
        self.assertEqual(sample_count, 32)

    def test_set_data_file_without_annotations(self):
        """
        Make sure setting data_file with no annotations registered does not
        raise any errors
        """
        meta = SigMFFile(TEST_METADATA)
        meta._metadata[SigMFFile.ANNOTATION_KEY].clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path_data = Path(tmpdir) / "datafile"
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            meta.set_data_file(temp_path_data)
            samples = meta.read_samples()
            self.assertTrue(len(samples) == 16)

    def test_set_data_file_with_annotations(self):
        """
        Make sure setting data_file with annotations registered use sample
        count from data_file and issue a warning if annotations have end
        indices bigger than file end index
        """
        meta = SigMFFile(TEST_METADATA)
        meta.add_annotation(start_index=0, length=32)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path_data = Path(tmpdir) / "datafile"
            TEST_FLOAT32_DATA.tofile(temp_path_data)
            with self.assertWarns(Warning):
                # Issues warning since file ends before the final annotatio
                meta.set_data_file(temp_path_data)
                samples = meta.read_samples()
                self.assertTrue(len(samples) == 16)


class TestMultichannel(unittest.TestCase):
    def setUp(self):
        """
        In order to check shapes we need some positive number of samples to work with.
        Number of samples should be lowest common factor of num_channels.
        """
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
                            sigmf.DATATYPE_KEY: f"{complex_prefix}{key}_le",
                            sigmf.NUM_CHANNELS_KEY: num_channels,
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
                sigmf.DATATYPE_KEY: "cu16_le",
                sigmf.NUM_CHANNELS_KEY: 3,
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
    meta = SigMFFile()
    top_sort_order = ["global", "captures", "annotations"]
    for kdx, key in enumerate(meta.ordered_metadata()):
        assert kdx == top_sort_order.index(key)


class TestBasicFunctionality(unittest.TestCase):
    """test basic SigMFFile functionality"""

    def test_default_constructor(self):
        """test default constructor"""
        SigMFFile()

    def test_set_non_required_global_field(self):
        """test setting field not in schema"""
        meta = SigMFFile()
        meta.set_global_field("this_is:not_in_the_schema", None)

    def test_add_annotation(self):
        """test basic annotation addition"""
        meta = SigMFFile()
        meta.add_capture(start_index=0)
        annot = {"latitude": 40.0, "longitude": -105.0}
        meta.add_annotation(start_index=0, length=128, metadata=annot)

    def test_load_from_archive(self):
        """test loading from archive"""
        with tempfile.NamedTemporaryFile(suffix=".sigmf") as temp_file:
            # create temporary data file
            with tempfile.NamedTemporaryFile(suffix=".sigmf-data", delete=False) as data_file:
                TEST_FLOAT32_DATA.tofile(data_file.name)
                meta = SigMFFile(TEST_METADATA, data_file=data_file.name)
            archive_path = meta.archive(name=temp_file.name, overwrite=True)
            loopback = sigmf.fromarchive(archive_path=archive_path)
            self.assertEqual(loopback._metadata, meta._metadata)


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
        alt_sigmf.set_global_field(sigmf.DATATYPE_KEY, "rf32_le")
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
        self.sigmf_obj.tofile(self.test_archive_path)
        with self.assertRaises(error.SigMFFileError) as context:
            self.sigmf_obj.tofile(self.test_archive_path, overwrite=False)
        self.assertIn("already exists", str(context.exception))

    def test_archive_overwrite_works(self):
        """tofile archive succeeds when archive exists and overwrite=True"""
        # create existing archive
        self.sigmf_obj.tofile(self.test_archive_path)
        self.assertTrue(self.test_archive_path.exists())
        original_checksum = self.sigmf_obj.get_global_field("core:sha512")

        # create sigmf object with different data
        alt_sigmf = SigMFFile()
        alt_sigmf.set_global_field(sigmf.DATATYPE_KEY, "rf32_le")
        alt_sigmf.set_global_field("core:description", "overwritten archive")
        alt_sigmf.set_data_file(self.alt_data_path)

        # should succeed with overwrite=True and content should change
        alt_sigmf.tofile(self.test_archive_path, overwrite=True)
        self.assertTrue(self.test_archive_path.exists())

        # verify by reading the archive content back
        loopback_sigmf = sigmf.fromarchive(self.test_archive_path)
        new_checksum = loopback_sigmf.get_global_field("core:sha512")

        self.assertEqual(loopback_sigmf.get_global_field("core:description"), "overwritten archive")
        self.assertNotEqual(original_checksum, new_checksum, "SHA512 checksum should change when overwritten")

    def test_default_behavior(self):
        """overwrite defaults to False for safety"""
        # create existing files
        self.sigmf_obj.tofile(self.test_meta_path)
        self.sigmf_obj.tofile(self.test_archive_path)

        # should raise exceptions with default overwrite=False
        with self.assertRaises(error.SigMFFileError):
            self.sigmf_obj.tofile(self.test_meta_path)

        with self.assertRaises(error.SigMFFileError):
            self.sigmf_obj.tofile(self.test_archive_path)


class TestFromarrayConvenience(unittest.TestCase):
    """Tests for the sigmf.fromarray() convenience function."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_basic_creation(self):
        """test creating SigMFFile from array"""
        meta = sigmf.fromarray(TEST_FLOAT32_DATA)
        self.assertEqual(meta.get_global_field(sigmf.DATATYPE_KEY), "rf32_le")
        np.testing.assert_array_equal(TEST_FLOAT32_DATA, meta[:])

    def test_write_separate_files(self):
        """test writing to separate meta and data files"""
        meta = sigmf.fromarray(TEST_FLOAT32_DATA)
        path = self.temp_dir / "basic"
        meta.tofile(str(path))
        self.assertTrue((self.temp_dir / "basic.sigmf-data").exists())
        self.assertTrue((self.temp_dir / "basic.sigmf-meta").exists())
        loopback = sigmf.fromfile(str(path))
        np.testing.assert_array_equal(TEST_FLOAT32_DATA, loopback[:])

    def test_write_archive(self):
        """test writing to uncompressed archive"""
        meta = sigmf.fromarray(TEST_FLOAT32_DATA)
        path = self.temp_dir / "archived.sigmf"
        meta.tofile(str(path))
        self.assertTrue((self.temp_dir / "archived.sigmf").exists())
        self.assertFalse((self.temp_dir / "archived.sigmf-data").exists())
        self.assertFalse((self.temp_dir / "archived.sigmf-meta").exists())
        loopback = sigmf.fromfile(str(path))
        np.testing.assert_array_equal(TEST_FLOAT32_DATA, loopback[:])

    def test_write_compressed_archive(self):
        """test writing to compressed archive"""
        meta = sigmf.fromarray(TEST_FLOAT32_DATA)
        path = self.temp_dir / "comp.sigmf.xz"
        meta.tofile(str(path))
        self.assertTrue((self.temp_dir / "comp.sigmf.xz").exists())
        self.assertFalse((self.temp_dir / "comp.sigmf-data").exists())
        self.assertFalse((self.temp_dir / "comp.sigmf-meta").exists())
        loopback = sigmf.fromfile(str(path))
        np.testing.assert_array_equal(TEST_FLOAT32_DATA, loopback[:])

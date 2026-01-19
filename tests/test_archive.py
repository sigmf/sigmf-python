# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for SigMFArchive"""

import codecs
import copy
import json
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np

from sigmf import SigMFFile, __specification__, error, fromfile
from sigmf.archive import SIGMF_DATASET_EXT, SIGMF_METADATA_EXT

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


class TestSigMFArchive(unittest.TestCase):
    """Tests for SigMF Archive functionality"""

    def setUp(self):
        """Create temporary directory and test SigMFFile"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_path_data = self.temp_dir / "trash.sigmf-data"
        self.temp_path_meta = self.temp_dir / "trash.sigmf-meta"
        self.temp_path_archive = self.temp_dir / "test.sigmf"
        TEST_FLOAT32_DATA.tofile(self.temp_path_data)
        self.sigmf_object = SigMFFile(copy.deepcopy(TEST_METADATA), data_file=self.temp_path_data)
        self.sigmf_object.tofile(self.temp_path_meta)
        self.sigmf_object.tofile(self.temp_path_archive, toarchive=True)
        self.sigmf_tarfile = tarfile.open(self.temp_path_archive, mode="r", format=tarfile.PAX_FORMAT)

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_archive_creation_requires_data_file(self):
        """Test that archiving without data file raises error"""
        self.sigmf_object.data_file = None
        with self.assertRaises(error.SigMFFileError):
            self.sigmf_object.archive(name=self.temp_path_archive)

    def test_archive_creation_validates_metadata(self):
        """Test that invalid metadata raises error"""
        del self.sigmf_object._metadata["global"]["core:datatype"]  # required field
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            self.sigmf_object.archive(name=self.temp_path_archive)

    def test_archive_creation_validates_extension(self):
        """Test that wrong extension raises error"""
        wrong_name = self.temp_dir / "temp_archive.zip"
        with self.assertRaises(error.SigMFFileError):
            self.sigmf_object.archive(name=wrong_name)

    def test_fileobj_ignores_extension(self):
        """Test that file object extension is ignored"""
        temp_archive_tar = self.temp_dir / "test.sigmf.tar"
        with open(temp_archive_tar, "wb") as temp:
            self.sigmf_object.archive(fileobj=temp)

    def test_custom_name_overrides_fileobj_name(self):
        """Test that name is used in file object"""
        with open(self.temp_path_archive, "w+b") as temp:
            self.sigmf_object.archive(name="testarchive", fileobj=temp)
            temp.seek(0)  # rewind to beginning of file after writing
            sigmf_tarfile = tarfile.open(fileobj=temp, mode="r")
            basedir, file1, file2 = sigmf_tarfile.getmembers()
            self.assertEqual(basedir.name, "testarchive")
            self.assertEqual(Path(file1.name).stem, "testarchive")
            self.assertEqual(Path(file2.name).stem, "testarchive")

    def test_fileobj_remains_open_after_archive(self):
        """Test that file object is not closed after archiving"""
        with open(self.temp_path_archive, "wb") as temp:
            self.sigmf_object.archive(fileobj=temp)
            self.assertFalse(temp.closed)

    def test_readonly_fileobj_raises_error(self):
        """Test that unwritable file object raises error"""
        temp_path = self.temp_dir / "temp_archive.sigmf"
        temp_path.touch()
        with open(temp_path, "rb") as temp:
            with self.assertRaises(error.SigMFFileError):
                self.sigmf_object.archive(fileobj=temp)

    def test_invalid_path_raises_error(self):
        """Test that unwritable name raises error"""
        # Cannot assume /root/ is unwritable (e.g. Docker environment)
        # so use invalid filename
        unwritable_file = "/bad_name/"
        with self.assertRaises(error.SigMFFileError):
            self.sigmf_object.archive(name=unwritable_file)

    def test_archive_contains_directory_and_files(self):
        """Test archive layout structure"""
        basedir, file1, file2 = self.sigmf_tarfile.getmembers()
        self.assertTrue(tarfile.TarInfo.isdir(basedir))
        self.assertTrue(tarfile.TarInfo.isfile(file1))
        self.assertTrue(tarfile.TarInfo.isfile(file2))

    def test_archive_files_have_correct_names_and_extensions(self):
        """Test tarfile names and extensions"""
        basedir, file1, file2 = self.sigmf_tarfile.getmembers()
        archive_name = basedir.name
        self.assertEqual(archive_name, Path(self.temp_path_archive).stem)
        file_extensions = {SIGMF_DATASET_EXT, SIGMF_METADATA_EXT}

        file1_name, file1_ext = Path(file1.name).stem, Path(file1.name).suffix
        self.assertEqual(file1_name, archive_name)
        self.assertIn(file1_ext, file_extensions)

        file_extensions.remove(file1_ext)

        file2_name, file2_ext = Path(file2.name).stem, Path(file2.name).suffix
        self.assertEqual(file2_name, archive_name)
        self.assertIn(file2_ext, file_extensions)

    def test_archive_files_have_correct_permissions(self):
        """Test tarfile permissions"""
        basedir, file1, file2 = self.sigmf_tarfile.getmembers()
        self.assertEqual(basedir.mode, 0o755)
        self.assertEqual(file1.mode, 0o644)
        self.assertEqual(file2.mode, 0o644)

    def test_archive_contents_match_original_data(self):
        """Test archive contents"""
        _, file1, file2 = self.sigmf_tarfile.getmembers()
        if file1.name.endswith(SIGMF_METADATA_EXT):
            mdfile = file1
            datfile = file2
        else:
            mdfile = file2
            datfile = file1

        bytestream_reader = codecs.getreader("utf-8")  # bytes -> str
        mdfile_reader = bytestream_reader(self.sigmf_tarfile.extractfile(mdfile))
        self.assertEqual(json.load(mdfile_reader), TEST_METADATA)

        datfile_reader = self.sigmf_tarfile.extractfile(datfile)
        # calling `fileno` on `tarfile.ExFileObject` throws error (?), but
        # np.fromfile requires it, so we need this extra step
        data = np.frombuffer(datfile_reader.read(), dtype=np.float32)

        np.testing.assert_array_equal(data, TEST_FLOAT32_DATA)

    def test_tarfile_format(self):
        """Tar file format is PAX"""
        self.assertEqual(self.sigmf_tarfile.format, tarfile.PAX_FORMAT)

    def test_archive_read_samples(self):
        """test that read_samples works correctly with archived data"""
        # load from archive
        archive_mdfile = fromfile(self.temp_path_archive)

        # verify sample count matches
        expected_sample_count = len(self.sigmf_object)
        self.assertEqual(archive_mdfile.sample_count, expected_sample_count)

        # verify read_samples returns same as slice
        samples_orig = TEST_FLOAT32_DATA[3:13]
        samples_read = archive_mdfile.read_samples(start_index=3, count=10)
        samples_sliced = archive_mdfile[3:13]
        np.testing.assert_array_equal(samples_orig, samples_sliced)
        np.testing.assert_array_equal(samples_orig, samples_read)

    def test_archive_read_samples_beyond_end(self):
        """test that read_samples beyond end of data raises error"""
        meta = fromfile(self.temp_path_archive)
        # FIXME: Should this raise a SigMFFileError instead?
        with self.assertRaises(OSError):
            meta.read_samples(start_index=meta.sample_count + 10, count=5)

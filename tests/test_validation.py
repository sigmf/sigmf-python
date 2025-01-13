# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Validator"""

import tempfile
import unittest
from pathlib import Path

from jsonschema.exceptions import ValidationError

import sigmf
from sigmf import SigMFFile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


def test_valid_data():
    """assure the supplied metadata is OK"""
    invalid_metadata = dict(TEST_METADATA)
    SigMFFile(TEST_METADATA).validate()


class CommandLineValidator(unittest.TestCase):
    """Check behavior of command-line parser"""

    def setUp(self):
        """Create a directory with some valid files"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = tmp_path = Path(self.tmp_dir.name)
        junk_path = tmp_path / "junk"
        TEST_FLOAT32_DATA.tofile(junk_path)
        some_meta = SigMFFile(TEST_METADATA, data_file=junk_path)
        some_meta.tofile(tmp_path / "a")
        some_meta.tofile(tmp_path / "b")
        some_meta.tofile(tmp_path / "c", toarchive=True)

    def tearDown(self):
        """cleanup"""
        self.tmp_dir.cleanup()

    def test_normal(self):
        """able to parse archives and non-archives"""
        args = (str(self.tmp_path / "*.sigmf*"),)
        sigmf.validate.main(args)

    def test_normal_skip(self):
        """able to skip checksum"""
        args = (str(self.tmp_path / "*.sigmf*"), "--skip-checksum")
        sigmf.validate.main(args)

    def test_partial(self):
        """checks some but not all files"""
        args = (str(self.tmp_path / "*"),)
        with self.assertRaises(SystemExit):
            sigmf.validate.main(args)

    def test_missing(self):
        """exit with rc=1 when run on empty"""
        with self.assertRaises(SystemExit) as cm:
            sigmf.validate.main(tuple())
        self.assertEqual((1,), cm.exception.args)

    def test_version(self):
        """exit with rc=0 after printing version"""
        args = ("--version",)
        with self.assertRaises(SystemExit) as cm:
            sigmf.validate.main(args)
        self.assertEqual((0,), cm.exception.args)


class FailingCases(unittest.TestCase):
    """Cases where the validator should throw an exception."""

    def setUp(self):
        self.metadata = dict(TEST_METADATA)

    def test_no_version(self):
        """core:version must be present"""
        del self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.VERSION_KEY]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_extra_top_level_key(self):
        """no extra keys allowed on the top level"""
        self.metadata["extra"] = 0
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_type(self):
        """license key must be string"""
        self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.LICENSE_KEY] = 1
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_capture_order(self):
        """metadata must have captures in order"""
        self.metadata[SigMFFile.CAPTURE_KEY] = [{SigMFFile.START_INDEX_KEY: 10}, {SigMFFile.START_INDEX_KEY: 9}]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_annotation_order(self):
        """metadata must have annotations in order"""
        self.metadata[SigMFFile.ANNOTATION_KEY] = [
            {
                SigMFFile.START_INDEX_KEY: 2,
                SigMFFile.LENGTH_INDEX_KEY: 120000,
            },
            {
                SigMFFile.START_INDEX_KEY: 1,
                SigMFFile.LENGTH_INDEX_KEY: 120000,
            },
        ]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_annotation_without_sample_count(self):
        """annotation without length should be accepted"""
        self.metadata[SigMFFile.ANNOTATION_KEY] = [{SigMFFile.START_INDEX_KEY: 2}]
        SigMFFile(self.metadata).validate()

    def test_invalid_hash(self):
        _, temp_path = tempfile.mkstemp()
        TEST_FLOAT32_DATA.tofile(temp_path)
        self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.HASH_KEY] = "derp"
        with self.assertRaises(sigmf.error.SigMFFileError):
            SigMFFile(metadata=self.metadata, data_file=temp_path)

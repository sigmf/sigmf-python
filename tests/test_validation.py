# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Validator"""

import tempfile
import unittest

from jsonschema.exceptions import ValidationError

import sigmf
from sigmf import SigMFFile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


def test_valid_data():
    '''assure the supplied metadata is OK'''
    invalid_metadata = dict(TEST_METADATA)
    SigMFFile(TEST_METADATA).validate()

class FailingCases(unittest.TestCase):
    '''Cases where the validator should throw an exception.'''
    def setUp(self):
        self.metadata = dict(TEST_METADATA)

    def test_no_version(self):
        '''core:version must be present'''
        del self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.VERSION_KEY]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_extra_top_level_key(self):
        '''no extra keys allowed on the top level'''
        self.metadata['extra'] = 0
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_extra_top_level_key(self):
        '''label must be less than 20 chars'''
        self.metadata[SigMFFile.ANNOTATION_KEY][0][SigMFFile.LABEL_KEY] = 'a' * 21
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_type(self):
        '''license key must be string'''
        self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.LICENSE_KEY] = 1
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_capture_order(self):
        '''metadata must have captures in order'''
        self.metadata[SigMFFile.CAPTURE_KEY] = [
            {SigMFFile.START_INDEX_KEY: 10},
            {SigMFFile.START_INDEX_KEY: 9}
        ]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_invalid_annotation_order(self):
        '''metadata must have annotations in order'''
        self.metadata[SigMFFile.ANNOTATION_KEY] = [
            {
                SigMFFile.START_INDEX_KEY: 2,
                SigMFFile.LENGTH_INDEX_KEY: 120000,
            },
            {
                SigMFFile.START_INDEX_KEY: 1,
                SigMFFile.LENGTH_INDEX_KEY: 120000,
            }
        ]
        with self.assertRaises(ValidationError):
            SigMFFile(self.metadata).validate()

    def test_annotation_without_sample_count(self):
        '''annotation without length should be accepted'''
        self.metadata[SigMFFile.ANNOTATION_KEY] = [
            {
                SigMFFile.START_INDEX_KEY: 2
            }
        ]
        SigMFFile(self.metadata).validate()


    def test_invalid_hash(self):
        _, temp_path = tempfile.mkstemp()
        TEST_FLOAT32_DATA.tofile(temp_path)
        self.metadata[SigMFFile.GLOBAL_KEY][SigMFFile.HASH_KEY] = 'derp'
        with self.assertRaises(sigmf.error.SigMFFileError):
            SigMFFile(metadata=self.metadata, data_file=temp_path)

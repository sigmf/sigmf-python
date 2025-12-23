"""Tests for dynamic attribute access functionality."""

import copy
import unittest

import numpy as np

from sigmf import SigMFFile
from sigmf.error import SigMFAccessError

from .testdata import TEST_METADATA

SOME_LICENSE = "CC0-1.0"
SOME_RECORDER = "HackRF Pro"
SOME_DOI = "10.1000/182"


class TestDynamicAttributeAccess(unittest.TestCase):
    """Test dynamic attribute access for core global metadata fields."""

    def setUp(self):
        """create test sigmf file with some initial metadata"""
        self.meta = SigMFFile(copy.deepcopy(TEST_METADATA))

    def test_getter_existing_fields(self):
        """test attribute getters for existing core fields"""
        # test common core fields
        # self.assertEqual(self.meta.sample_rate, self.meta.get_global_field(SigMFFile.SAMPLE_RATE_KEY))
        # self.assertEqual(self.meta.author, self.meta.get_global_field(SigMFFile.AUTHOR_KEY))
        self.assertEqual(self.meta.datatype, self.meta.get_global_field(SigMFFile.DATATYPE_KEY))
        self.assertEqual(self.meta.sha512, self.meta.get_global_field(SigMFFile.HASH_KEY))
        # self.assertEqual(self.meta.description, self.meta.get_global_field(SigMFFile.DESCRIPTION_KEY))

    def test_getter_missing_core_fields(self):
        """test that getter raises SigMFAccessError for missing core fields"""
        with self.assertRaises(SigMFAccessError) as context:
            _ = self.meta.license
        self.assertIn(SigMFFile.LICENSE_KEY, str(context.exception))

    def test_getter_nonexistent_attribute(self):
        """test that getter raises AttributeError for non-core attributes"""
        with self.assertRaises(AttributeError) as context:
            _ = self.meta.nonexistent_field
        self.assertIn("nonexistent_field", str(context.exception))

    def test_setter_new_fields(self):
        """test that attribute setters work for new core fields"""
        # set various core global fields using attribute notation
        self.meta.license = SOME_LICENSE
        self.meta.meta_doi = SOME_DOI
        self.meta.recorder = SOME_RECORDER

        # verify they were set correctly
        self.assertEqual(self.meta.license, SOME_LICENSE)
        self.assertEqual(self.meta.meta_doi, SOME_DOI)
        self.assertEqual(self.meta.recorder, SOME_RECORDER)

    def test_setter_overwrite_fields(self):
        """test that attribute setters can overwrite existing fields"""
        self.meta.sha512 = "effec7"
        self.assertEqual(self.meta.sha512, "effec7")

    def test_setter_noncore_attributes(self):
        """test that setter works for non-core object attributes"""
        # set a regular attribute
        self.meta.custom_attribute = "test value"

        # verify it was set as a regular attribute
        self.assertEqual(self.meta.custom_attribute, "test value")

        # verify it doesn't appear in metadata
        self.assertNotIn("custom_attribute", self.meta.get_global_info())

    def test_method_vs_attribute_equivalence(self):
        """test that method-based and attribute-based access are equivalent"""
        # set via method, access via attribute
        self.meta.set_global_field(SigMFFile.LICENSE_KEY, SOME_LICENSE)
        self.assertEqual(self.meta.license, SOME_LICENSE)

        # set via attribute, access via method
        self.meta.recorder = SOME_RECORDER
        self.assertEqual(self.meta.get_global_field(SigMFFile.RECORDER_KEY), SOME_RECORDER)

    def test_private_attributes_unaffected(self):
        """test that private attributes work normally"""
        # private attributes should not trigger dynamic behavior
        self.meta._test_private = "private_value"
        self.assertEqual(self.meta._test_private, "private_value")

    def test_existing_properties_unaffected(self):
        """test that existing class properties work normally"""
        # test that existing properties like data_file still work
        self.meta.data_file = None  # this should work normally
        self.assertIsNone(self.meta.data_file)

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for collections"""

import copy
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sigmf import schema
from sigmf.archive import SIGMF_COLLECTION_EXT, SIGMF_DATASET_EXT, SIGMF_METADATA_EXT
from sigmf.error import SigMFFileError, SigMFFileExistsError
from sigmf.sigmffile import SigMFCollection, SigMFFile, fromfile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


class TestCollection(unittest.TestCase):
    """unit tests for colections"""

    def setUp(self):
        """create temporary path"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """remove temporary path"""
        shutil.rmtree(self.temp_dir)

    @given(st.sampled_from([".", "subdir/", "sub0/sub1/sub2/"]))
    def test_load_collection(self, subdir: str) -> None:
        """test path handling for collections"""
        data_name1 = "dat1" + SIGMF_DATASET_EXT
        data_name2 = "dat2" + SIGMF_DATASET_EXT
        meta_name1 = "dat1" + SIGMF_METADATA_EXT
        meta_name2 = "dat2" + SIGMF_METADATA_EXT
        collection_name = "collection" + SIGMF_COLLECTION_EXT
        data_path1 = self.temp_dir / subdir / data_name1
        data_path2 = self.temp_dir / subdir / data_name2
        meta_path1 = self.temp_dir / subdir / meta_name1
        meta_path2 = self.temp_dir / subdir / meta_name2
        collection_path = self.temp_dir / subdir / collection_name
        os.makedirs(collection_path.parent, exist_ok=True)

        # create data files
        TEST_FLOAT32_DATA.tofile(data_path1)
        TEST_FLOAT32_DATA.tofile(data_path2)

        # create metadata files
        metadata = copy.deepcopy(TEST_METADATA)
        meta1 = SigMFFile(metadata=metadata, data_file=data_path1)
        meta2 = SigMFFile(metadata=metadata, data_file=data_path2)
        meta1.tofile(meta_path1, overwrite=True)
        meta2.tofile(meta_path2, overwrite=True)

        # create collection
        collection = SigMFCollection(
            metafiles=[meta_name1, meta_name2],
            base_path=str(self.temp_dir / subdir),
        )
        collection.tofile(collection_path, overwrite=True)

        # load collection
        collection_loopback = fromfile(collection_path)
        meta1_loopback = collection_loopback.get_SigMFFile(stream_index=0)
        meta2_loopback = collection_loopback.get_SigMFFile(stream_index=1)

        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, meta1_loopback.read_samples()))
        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, meta2_loopback[:]))


class TestCollectionConstructor(unittest.TestCase):
    """tests for SigMFCollection constructor"""

    def test_empty_constructor(self):
        """test that a collection can be created with no arguments"""
        collection = SigMFCollection(skip_checksums=True)
        self.assertIsInstance(collection, SigMFCollection)
        self.assertEqual(len(collection), 0)
        self.assertEqual(collection.get_stream_names(), [])

    def test_constructor_with_metadata(self):
        """test that a collection can be created with a metadata dict"""
        from sigmf import __specification__

        metadata = {
            SigMFCollection.COLLECTION_KEY: {
                SigMFCollection.VERSION_KEY: __specification__,
                SigMFCollection.STREAMS_KEY: [],
            }
        }
        collection = SigMFCollection(metadata=metadata, skip_checksums=True)
        self.assertIsInstance(collection, SigMFCollection)
        self.assertEqual(len(collection), 0)


class TestCollectionRoundTrip(unittest.TestCase):
    """tests for SigMFCollection round-trip write/read"""

    def setUp(self):
        """create temporary directory and populate with SigMF files"""
        self.temp_dir = Path(tempfile.mkdtemp())
        # create two SigMF recordings
        meta_name1 = "stream0" + SIGMF_METADATA_EXT
        meta_name2 = "stream1" + SIGMF_METADATA_EXT
        data_path1 = self.temp_dir / ("stream0" + SIGMF_DATASET_EXT)
        data_path2 = self.temp_dir / ("stream1" + SIGMF_DATASET_EXT)
        TEST_FLOAT32_DATA.tofile(data_path1)
        TEST_FLOAT32_DATA.tofile(data_path2)
        meta1 = SigMFFile(metadata=copy.deepcopy(TEST_METADATA), data_file=data_path1)
        meta2 = SigMFFile(metadata=copy.deepcopy(TEST_METADATA), data_file=data_path2)
        meta1.tofile(self.temp_dir / meta_name1, overwrite=True)
        meta2.tofile(self.temp_dir / meta_name2, overwrite=True)
        self.meta_name1 = meta_name1
        self.meta_name2 = meta_name2
        self.collection_path = self.temp_dir / ("mycollection" + SIGMF_COLLECTION_EXT)

    def tearDown(self):
        """remove temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_round_trip_metadata(self):
        """test that collection metadata survives a write/read round-trip"""
        collection = SigMFCollection(
            metafiles=[self.meta_name1, self.meta_name2],
            base_path=str(self.temp_dir),
        )
        collection.set_collection_field(SigMFCollection.AUTHOR_KEY, "Round Trip Tester")
        collection.set_collection_field(SigMFCollection.DESCRIPTION_KEY, "A round-trip test collection")
        collection.set_collection_field(SigMFCollection.LICENSE_KEY, "https://creativecommons.org/licenses/by-sa/4.0/")

        collection.tofile(self.collection_path)

        # read back
        collection_rt = fromfile(self.collection_path)

        self.assertIsInstance(collection_rt, SigMFCollection)
        self.assertEqual(len(collection_rt), 2)
        self.assertEqual(collection_rt.get_stream_names(), ["stream0", "stream1"])
        self.assertEqual(collection_rt.get_collection_field(SigMFCollection.AUTHOR_KEY), "Round Trip Tester")
        self.assertEqual(
            collection_rt.get_collection_field(SigMFCollection.DESCRIPTION_KEY), "A round-trip test collection"
        )
        self.assertEqual(
            collection_rt.get_collection_field(SigMFCollection.LICENSE_KEY),
            "https://creativecommons.org/licenses/by-sa/4.0/",
        )

    def test_round_trip_collection_info(self):
        """test that get_collection_info returns a dict matching what was set"""
        collection = SigMFCollection(
            metafiles=[self.meta_name1, self.meta_name2],
            base_path=str(self.temp_dir),
        )
        collection.set_collection_field(SigMFCollection.AUTHOR_KEY, "Test Author")
        collection.tofile(self.collection_path)

        collection_rt = fromfile(self.collection_path)
        info = collection_rt.get_collection_info()
        self.assertIsInstance(info, dict)
        self.assertIn(SigMFCollection.AUTHOR_KEY, info)
        self.assertEqual(info[SigMFCollection.AUTHOR_KEY], "Test Author")
        self.assertIn(SigMFCollection.VERSION_KEY, info)
        self.assertIn(SigMFCollection.STREAMS_KEY, info)

    def test_round_trip_json_content(self):
        """test that the written collection file is valid JSON with expected structure"""
        collection = SigMFCollection(
            metafiles=[self.meta_name1, self.meta_name2],
            base_path=str(self.temp_dir),
        )
        collection.tofile(self.collection_path)

        with open(self.collection_path, "r") as f:
            data = json.load(f)

        self.assertIn(SigMFCollection.COLLECTION_KEY, data)
        self.assertIn(SigMFCollection.STREAMS_KEY, data[SigMFCollection.COLLECTION_KEY])
        self.assertIn(SigMFCollection.VERSION_KEY, data[SigMFCollection.COLLECTION_KEY])
        streams = data[SigMFCollection.COLLECTION_KEY][SigMFCollection.STREAMS_KEY]
        self.assertEqual(len(streams), 2)
        for stream in streams:
            self.assertIn("name", stream)
            self.assertIn("hash", stream)


class TestCollectionValidation(unittest.TestCase):
    """tests for SigMFCollection validation against the JSON schema"""

    def _validate(self, metadata):
        """helper: validate collection metadata against the collection schema"""
        col_schema = schema.get_schema(schema_file=schema.SCHEMA_COLLECTION)
        jsonschema.validators.validate(instance=metadata, schema=col_schema)

    def test_valid_empty_collection(self):
        """a minimal collection with only core:version should be schema-valid"""
        collection = SigMFCollection(skip_checksums=True)
        self._validate(collection._metadata)

    def test_valid_collection_with_optional_fields(self):
        """a collection with optional fields set should be schema-valid"""
        collection = SigMFCollection(skip_checksums=True)
        collection.set_collection_field(SigMFCollection.AUTHOR_KEY, "Test Author")
        collection.set_collection_field(SigMFCollection.DESCRIPTION_KEY, "Test description")
        collection.set_collection_field(SigMFCollection.LICENSE_KEY, "https://example.com/license")
        collection.set_collection_field(SigMFCollection.COLLECTION_DOI_KEY, "10.1000/xyz123")
        self._validate(collection._metadata)

    def test_invalid_collection_missing_version(self):
        """a collection missing core:version should fail schema validation"""
        metadata = {SigMFCollection.COLLECTION_KEY: {}}
        col_schema = schema.get_schema(schema_file=schema.SCHEMA_COLLECTION)
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            jsonschema.validators.validate(instance=metadata, schema=col_schema)

    def test_invalid_collection_missing_collection_key(self):
        """a metadata dict without the top-level 'collection' key should fail"""
        metadata = {}
        col_schema = schema.get_schema(schema_file=schema.SCHEMA_COLLECTION)
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            jsonschema.validators.validate(instance=metadata, schema=col_schema)

    def test_valid_collection_with_extensions(self):
        """a collection with a valid extensions array should be schema-valid"""
        collection = SigMFCollection(skip_checksums=True)
        collection.set_collection_field(
            SigMFCollection.EXTENSIONS_KEY,
            [{"name": "antenna", "version": "1.0.0", "optional": True}],
        )
        self._validate(collection._metadata)


class TestCollectionCommonUseCases(unittest.TestCase):
    """tests for common SigMFCollection use cases"""

    def setUp(self):
        """create temporary directory and two SigMF recordings"""
        self.temp_dir = Path(tempfile.mkdtemp())
        for name in ("rec0", "rec1", "rec2"):
            data_path = self.temp_dir / (name + SIGMF_DATASET_EXT)
            meta_path = self.temp_dir / (name + SIGMF_METADATA_EXT)
            TEST_FLOAT32_DATA.tofile(data_path)
            meta = SigMFFile(metadata=copy.deepcopy(TEST_METADATA), data_file=data_path)
            meta.tofile(meta_path, overwrite=True)
        self.metafiles = [f"{name}{SIGMF_METADATA_EXT}" for name in ("rec0", "rec1", "rec2")]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _make_collection(self, metafiles=None):
        """helper: create a SigMFCollection using the temp dir"""
        if metafiles is None:
            metafiles = self.metafiles
        return SigMFCollection(metafiles=metafiles, base_path=str(self.temp_dir))

    def test_len(self):
        """__len__ should return the number of streams"""
        collection = self._make_collection()
        self.assertEqual(len(collection), 3)

    def test_len_empty(self):
        """an empty collection should have length 0"""
        collection = SigMFCollection(skip_checksums=True)
        self.assertEqual(len(collection), 0)

    def test_get_stream_names(self):
        """get_stream_names should return base names in order"""
        collection = self._make_collection()
        names = collection.get_stream_names()
        self.assertEqual(names, ["rec0", "rec1", "rec2"])

    def test_get_sigmffile_by_index(self):
        """get_SigMFFile with stream_index should return correct SigMFFile"""
        collection = self._make_collection()
        sf = collection.get_SigMFFile(stream_index=0)
        self.assertIsInstance(sf, SigMFFile)
        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, sf.read_samples()))

    def test_get_sigmffile_by_name(self):
        """get_SigMFFile with stream_name should return correct SigMFFile"""
        collection = self._make_collection()
        sf = collection.get_SigMFFile(stream_name="rec1")
        self.assertIsInstance(sf, SigMFFile)
        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, sf.read_samples()))

    def test_get_sigmffile_invalid_name(self):
        """get_SigMFFile with an unknown stream_name should return None"""
        collection = self._make_collection()
        result = collection.get_SigMFFile(stream_name="nonexistent")
        self.assertIsNone(result)

    def test_set_get_collection_field(self):
        """set_collection_field and get_collection_field should round-trip values"""
        collection = self._make_collection()
        collection.set_collection_field(SigMFCollection.AUTHOR_KEY, "Jane Doe")
        self.assertEqual(collection.get_collection_field(SigMFCollection.AUTHOR_KEY), "Jane Doe")

    def test_get_collection_field_default(self):
        """get_collection_field should return default when key is absent"""
        collection = self._make_collection()
        result = collection.get_collection_field("core:nonexistent_key", default="fallback")
        self.assertEqual(result, "fallback")

    def test_set_get_collection_info(self):
        """set_collection_info and get_collection_info should round-trip a dict"""
        from sigmf import __specification__

        collection = self._make_collection()
        new_info = {
            SigMFCollection.VERSION_KEY: __specification__,
            SigMFCollection.AUTHOR_KEY: "Info Author",
            SigMFCollection.STREAMS_KEY: collection.get_collection_field(SigMFCollection.STREAMS_KEY),
        }
        collection.set_collection_info(new_info)
        info = collection.get_collection_info()
        self.assertEqual(info[SigMFCollection.AUTHOR_KEY], "Info Author")

    def test_overwrite_protection(self):
        """writing a collection to an existing file without overwrite=True should raise"""
        collection_path = self.temp_dir / ("test" + SIGMF_COLLECTION_EXT)
        collection = self._make_collection()
        collection.tofile(collection_path)
        with self.assertRaises(SigMFFileExistsError):
            collection.tofile(collection_path)

    def test_overwrite_allowed(self):
        """writing with overwrite=True should succeed even if file exists"""
        collection_path = self.temp_dir / ("test" + SIGMF_COLLECTION_EXT)
        collection = self._make_collection()
        collection.tofile(collection_path)
        collection.tofile(collection_path, overwrite=True)
        self.assertTrue(collection_path.exists())

    def test_skip_checksums(self):
        """skip_checksums=True should allow creating a collection without verifying hashes"""
        collection_path = self.temp_dir / ("test" + SIGMF_COLLECTION_EXT)
        collection = self._make_collection()
        collection.tofile(collection_path)
        # test via SigMFCollection constructor with skip_checksums=True
        with open(collection_path, "r") as f:
            metadata = json.load(f)
        collection_loaded = SigMFCollection(metadata=metadata, base_path=str(self.temp_dir), skip_checksums=True)
        self.assertIsInstance(collection_loaded, SigMFCollection)
        self.assertEqual(len(collection_loaded), 3)

    def test_verify_stream_hashes_valid(self):
        """verify_stream_hashes should not raise when hashes are correct"""
        collection = self._make_collection()
        # should not raise
        collection.verify_stream_hashes()

    def test_verify_stream_hashes_invalid(self):
        """verify_stream_hashes should raise when a stream hash is wrong"""
        collection = self._make_collection()
        # corrupt the hash of the first stream
        streams = collection.get_collection_field(SigMFCollection.STREAMS_KEY)
        streams[0]["hash"] = "badhash"
        collection.set_collection_field(SigMFCollection.STREAMS_KEY, streams)
        with self.assertRaises(SigMFFileError):
            collection.verify_stream_hashes()

    def test_error_on_nonexistent_metafile(self):
        """constructing a collection with a non-existent file should raise SigMFFileError"""
        with self.assertRaises(SigMFFileError):
            SigMFCollection(
                metafiles=["does_not_exist" + SIGMF_METADATA_EXT],
                base_path=str(self.temp_dir),
            )

    def test_error_on_non_meta_extension(self):
        """constructing a collection with a file lacking .sigmf-meta extension should raise"""
        with self.assertRaises(SigMFFileError):
            SigMFCollection(
                metafiles=["rec0" + SIGMF_DATASET_EXT],
                base_path=str(self.temp_dir),
            )

    def test_set_streams_updates_hashes(self):
        """set_streams should recompute hashes for the specified metafiles"""
        collection = self._make_collection(metafiles=["rec0" + SIGMF_METADATA_EXT])
        self.assertEqual(len(collection), 1)
        # add more streams
        collection.set_streams(["rec0" + SIGMF_METADATA_EXT, "rec1" + SIGMF_METADATA_EXT])
        self.assertEqual(len(collection), 2)
        names = collection.get_stream_names()
        self.assertIn("rec0", names)
        self.assertIn("rec1", names)

    def test_collection_dumps_is_valid_json(self):
        """dumps() should produce valid JSON containing collection data"""
        collection = self._make_collection()
        s = collection.dumps()
        data = json.loads(s)
        self.assertIn(SigMFCollection.COLLECTION_KEY, data)
        self.assertIn(SigMFCollection.STREAMS_KEY, data[SigMFCollection.COLLECTION_KEY])
        self.assertEqual(len(data[SigMFCollection.COLLECTION_KEY][SigMFCollection.STREAMS_KEY]), 3)

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for collections"""

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sigmf.archive import SIGMF_COLLECTION_EXT, SIGMF_DATASET_EXT, SIGMF_METADATA_EXT
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

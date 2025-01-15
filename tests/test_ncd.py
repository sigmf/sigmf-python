# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Non-Conforming Datasets"""

import copy
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from sigmf.error import SigMFFileError
from sigmf.sigmffile import SigMFFile, fromfile

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


class TestNonConformingDataset(unittest.TestCase):
    """unit tests for NCD"""

    def setUp(self):
        """create temporary path"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """remove temporary path"""
        shutil.rmtree(self.temp_dir)

    @given(st.sampled_from([".", "subdir/", "sub0/sub1/sub2/"]))
    def test_load_ncd(self, subdir: str) -> None:
        """test loading non-conforming dataset"""
        data_path = self.temp_dir / subdir / "dat.bin"
        meta_path = self.temp_dir / subdir / "dat.sigmf-meta"
        Path.mkdir(data_path.parent, parents=True, exist_ok=True)

        # create data file
        TEST_FLOAT32_DATA.tofile(data_path)

        # create metadata file
        ncd_metadata = copy.deepcopy(TEST_METADATA)
        meta = SigMFFile(metadata=ncd_metadata, data_file=data_path)
        meta.tofile(meta_path)

        # load dataset & validate we can read all the data
        meta_loopback = fromfile(meta_path)
        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, meta_loopback.read_samples()))
        self.assertTrue(np.array_equal(TEST_FLOAT32_DATA, meta_loopback[:]))

        # delete the non-conforming dataset and ensure error is raised due to missing dataset;
        # in Windows the SigMFFile instances need to be garbage collected first,
        # otherwise the np.memmap instances (stored in self._memmap) block the deletion
        meta = None
        meta_loopback = None
        Path.unlink(data_path)
        with self.assertRaises(SigMFFileError):
            _ = fromfile(meta_path)

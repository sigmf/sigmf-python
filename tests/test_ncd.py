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
import warnings
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
        meta.tofile(meta_path, overwrite=True)

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

    def test_ncd_priority_over_conforming_dataset(self) -> None:
        """test that NCD file specified in core:dataset is prioritized over .sigmf-data file"""
        base_name = "conflicting_dataset"
        meta_path = self.temp_dir / f"{base_name}.sigmf-meta"
        ncd_path = self.temp_dir / f"{base_name}.fleeb"
        conforming_path = self.temp_dir / f"{base_name}.sigmf-data"

        # create two different datasets with distinct data for verification
        ncd_data = np.array([100, 200, 300, 400], dtype=np.float32)
        conforming_data = np.array([1, 2, 3, 4], dtype=np.float32)

        # write both data files
        ncd_data.tofile(ncd_path)
        conforming_data.tofile(conforming_path)

        # create metadata that references the ncd file
        ncd_metadata = copy.deepcopy(TEST_METADATA)
        ncd_metadata[SigMFFile.GLOBAL_KEY][SigMFFile.DATASET_KEY] = f"{base_name}.fleeb"
        ncd_metadata[SigMFFile.GLOBAL_KEY][SigMFFile.NUM_CHANNELS_KEY] = 1
        ncd_metadata[SigMFFile.GLOBAL_KEY][SigMFFile.DATATYPE_KEY] = "rf32_le"
        ncd_metadata[SigMFFile.GLOBAL_KEY].pop(SigMFFile.SHA512_KEY, None)
        ncd_metadata[SigMFFile.ANNOTATION_KEY] = [{SigMFFile.SAMPLE_COUNT_KEY: 4, SigMFFile.SAMPLE_START_KEY: 0}]

        # write metadata file
        meta = SigMFFile(metadata=ncd_metadata)
        meta.tofile(meta_path, overwrite=True)

        # verify warning is generated about conflicting datasets
        with self.assertWarns(UserWarning):
            loaded_meta = fromfile(meta_path)

        # verify that the ncd data is loaded, not the conforming data
        loaded_data = loaded_meta.read_samples()
        self.assertTrue(np.array_equal(ncd_data, loaded_data), "NCD file should be prioritized over .sigmf-data")

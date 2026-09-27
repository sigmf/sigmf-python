# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Non-Conforming Datasets"""

import copy
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

import sigmf
from sigmf.error import SigMFFileError
from sigmf.sigmffile import SigMFFile, fromfile

from .conftest import TEST_FLOAT32_DATA, TEST_METADATA


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
        ncd_metadata = TEST_METADATA
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
        ncd_metadata[SigMFFile.GLOBAL_KEY][sigmf.DATASET_KEY] = f"{base_name}.fleeb"
        ncd_metadata[SigMFFile.GLOBAL_KEY][sigmf.NUM_CHANNELS_KEY] = 1
        ncd_metadata[SigMFFile.GLOBAL_KEY][sigmf.DATATYPE_KEY] = "rf32_le"
        ncd_metadata[SigMFFile.GLOBAL_KEY].pop(sigmf.SHA512_KEY, None)
        ncd_metadata[SigMFFile.ANNOTATION_KEY] = [{sigmf.SAMPLE_COUNT_KEY: 4, sigmf.SAMPLE_START_KEY: 0}]

        # write metadata file
        meta = SigMFFile(metadata=ncd_metadata)
        meta.tofile(meta_path, overwrite=True)

        # verify warning is generated about conflicting datasets
        with self.assertWarns(UserWarning):
            loaded_meta = fromfile(meta_path)

        # verify that the ncd data is loaded, not the conforming data
        loaded_data = loaded_meta.read_samples()
        self.assertTrue(np.array_equal(ncd_data, loaded_data), "NCD file should be prioritized over .sigmf-data")


class TestHeaderFooter(unittest.TestCase):
    """Look for quirks in NCD related to header and trailing bytes"""

    # header/trailing byte counts to exercise
    byte_strategy = st.integers(min_value=1, max_value=4096)

    def setUp(self):
        """create temporary path"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.samples = TEST_FLOAT32_DATA

    def tearDown(self):
        """remove temporary path"""
        shutil.rmtree(self.temp_dir)

    def _make_ncd(self, header_bytes: int, trailing_bytes: int, data_file_first: bool = True) -> SigMFFile:
        """
        Write an NCD with the given header/trailing bytes, returning its SigMFFile.

        `data_file_first` selects whether `set_data_file` is called before
        `add_capture` (the order used by the converters) or after.
        """
        ncd_path = Path(tempfile.mkdtemp(dir=self.temp_dir)) / "dat.bin"
        with open(ncd_path, "wb") as handle:
            handle.write(b"\x00" * header_bytes)
            handle.write(self.samples.tobytes())
            handle.write(b"\xff" * trailing_bytes)
        global_info = {
            sigmf.DATATYPE_KEY: "rf32_le",
            sigmf.NUM_CHANNELS_KEY: 1,
            sigmf.TRAILING_BYTES_KEY: trailing_bytes,
            sigmf.DATASET_KEY: ncd_path.name,
        }
        capture = {sigmf.HEADER_BYTES_KEY: header_bytes}
        meta = SigMFFile(global_info=global_info)
        if data_file_first:
            meta.set_data_file(data_file=ncd_path, offset=header_bytes)
            meta.add_capture(0, metadata=capture)
        else:
            meta.add_capture(0, metadata=capture)
            meta.set_data_file(data_file=ncd_path, offset=header_bytes)
        return meta

    @given(header_bytes=byte_strategy, trailing_bytes=byte_strategy)
    def test_read(self, header_bytes: int, trailing_bytes: int) -> None:
        """header/trailing bytes must not be exposed as samples"""
        for data_file_first in (True, False):
            meta = self._make_ncd(header_bytes, trailing_bytes, data_file_first=data_file_first)
            self.assertEqual(len(self.samples), meta.sample_count)
            self.assertEqual(len(self.samples), len(meta))
            np.testing.assert_array_equal(self.samples, meta.read_samples())

    @given(header_bytes=byte_strategy, trailing_bytes=byte_strategy)
    def test_archive_roundtrip(self, header_bytes: int, trailing_bytes: int) -> None:
        """archiving an NCD stores the original file and round-trips"""
        meta = self._make_ncd(header_bytes, trailing_bytes)
        archive_path = Path(meta.data_file).parent / "ncd.sigmf"
        meta.tofile(archive_path)

        # the archive must carry the NCD under its original name, not .sigmf-data
        with tarfile.open(archive_path) as tar:
            names = tar.getnames()
        self.assertIn("ncd/dat.bin", names)
        self.assertNotIn("ncd/ncd.sigmf-data", names)

        # round-trip must recover exactly the original samples
        loopback = fromfile(archive_path)
        np.testing.assert_array_equal(self.samples, loopback.read_samples())

    def test_oversized_trailing_bytes(self) -> None:
        """metadata that skips more bytes than the dataset holds must raise"""
        ncd_path = self.temp_dir / "dat.bin"
        self.samples.tofile(ncd_path)
        meta = SigMFFile(
            global_info={
                sigmf.DATATYPE_KEY: "rf32_le",
                sigmf.NUM_CHANNELS_KEY: 1,
                sigmf.TRAILING_BYTES_KEY: self.samples.nbytes + 1,
                sigmf.DATASET_KEY: ncd_path.name,
            }
        )
        with self.assertRaises(SigMFFileError):
            meta.set_data_file(data_file=ncd_path, offset=0)

    def test_oversized_header_bytes(self) -> None:
        """adding a capture whose header_bytes exceeds the dataset must raise"""
        ncd_path = self.temp_dir / "dat.bin"
        self.samples.tofile(ncd_path)
        meta = SigMFFile(
            global_info={
                sigmf.DATATYPE_KEY: "rf32_le",
                sigmf.NUM_CHANNELS_KEY: 1,
                sigmf.DATASET_KEY: ncd_path.name,
            }
        )
        meta.set_data_file(data_file=ncd_path, offset=0)
        with self.assertRaises(SigMFFileError):
            meta.add_capture(0, metadata={sigmf.HEADER_BYTES_KEY: self.samples.nbytes + 1})

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for BLUE Converter"""

import tempfile
import unittest
from pathlib import Path
from typing import cast

import sigmf
from sigmf.convert.blue import blue_to_sigmf

from .testdata import NONSIGMF_ENV, NONSIGMF_REPO


class TestBlueConverter(unittest.TestCase):
    """BLUE converter tests using external files"""

    def setUp(self) -> None:
        """setup paths to blue files"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        if not NONSIGMF_REPO:
            # skip test if environment variable not set
            self.skipTest(f"Set {NONSIGMF_ENV} environment variable to path with BLUE files to run test.")

        # look for blue files in blue/ directory
        blue_dir = NONSIGMF_REPO / "blue"
        self.bluefiles = []
        if blue_dir.exists():
            for ext in ["*.cdif", "*.tmp"]:
                self.bluefiles.extend(blue_dir.glob(f"**/{ext}"))

        if not self.bluefiles:
            self.fail(f"No BLUE files (*.cdif, *.tmp) found in {blue_dir}.")

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def _validate_ncd_structure(self, meta, expected_file):
        """validate basic NCD structure"""
        self.assertEqual(meta.data_file, expected_file, "NCD should point to original file")
        self.assertIn("core:trailing_bytes", meta._metadata["global"])
        captures = meta.get_captures()
        self.assertGreater(len(captures), 0, "Should have at least one capture")
        self.assertIn("core:header_bytes", captures[0])

        # validate SigMF spec compliance: NCDs must not have metadata_only field
        global_meta = meta._metadata["global"]
        has_dataset = "core:dataset" in global_meta
        has_metadata_only = "core:metadata_only" in global_meta

        self.assertTrue(has_dataset, "NCD should have core:dataset field")
        self.assertFalse(has_metadata_only, "NCD should NOT have core:metadata_only field (spec violation)")

        return captures

    def _validate_auto_detection(self, file_path):
        """validate auto-detection works and returns valid NCD"""
        meta_auto_raw = sigmf.fromfile(file_path)
        # auto-detection should return SigMFFile, not SigMFCollection
        self.assertIsInstance(meta_auto_raw, sigmf.SigMFFile)
        meta_auto = cast(sigmf.SigMFFile, meta_auto_raw)
        # data_file might be Path or str, so convert both for comparison
        self.assertEqual(str(meta_auto.data_file), str(file_path))
        self.assertIn("core:trailing_bytes", meta_auto._metadata["global"])
        return meta_auto

    def test_blue_to_sigmf_pair(self):
        """test standard blue to sigmf conversion with file pairs"""
        for bluefile in self.bluefiles:
            sigmf_path = self.tmp_path / bluefile.stem
            meta = blue_to_sigmf(blue_path=bluefile, out_path=sigmf_path)
            if not meta.get_global_field("core:metadata_only"):
                meta.read_samples(count=10)
            self.assertIsInstance(meta, sigmf.SigMFFile)

    def test_blue_to_sigmf_archive(self):
        """test blue to sigmf conversion with archive output"""
        for bluefile in self.bluefiles:
            sigmf_path = self.tmp_path / f"{bluefile.stem}_archive"
            meta = blue_to_sigmf(blue_path=bluefile, out_path=str(sigmf_path), create_archive=True)
            self.assertIsInstance(meta, sigmf.SigMFFile)

    def test_blue_to_sigmf_ncd(self):
        """test blue to sigmf conversion as Non-Conforming Dataset"""
        for bluefile in self.bluefiles:
            meta = blue_to_sigmf(blue_path=str(bluefile), create_ncd=True)

            # validate basic NCD structure
            self._validate_ncd_structure(meta, bluefile)

            # verify this is metadata-only (no separate data file created)
            self.assertIsInstance(meta.data_buffer, type(meta.data_buffer))

            # test that data can be read if not metadata-only
            if not meta.get_global_field("core:metadata_only"):
                _ = meta.read_samples(count=10)

    def test_blue_auto_detection(self):
        """test automatic BLUE detection through fromfile()"""
        for bluefile in self.bluefiles:
            # validate auto-detection works
            self._validate_auto_detection(bluefile)

    def test_blue_directory_files_ncd(self):
        """test NCD conversion"""
        for blue_file in self.bluefiles:
            meta = blue_to_sigmf(blue_path=str(blue_file), create_ncd=True)

            # validate basic NCD structure
            self._validate_ncd_structure(meta, blue_file)

            # validate auto-detection also works
            self._validate_auto_detection(blue_file)

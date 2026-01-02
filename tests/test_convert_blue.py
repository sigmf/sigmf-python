# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for BLUE Converter"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sigmf
from sigmf.convert.blue import blue_to_sigmf

from .test_convert_wav import _validate_ncd
from .testdata import NONSIGMF_ENV, NONSIGMF_REPO


class TestBlueWithNonSigMFRepo(unittest.TestCase):
    """BLUE converter tests using external files"""

    def setUp(self) -> None:
        """setup paths to blue files"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        if not NONSIGMF_REPO:
            # skip test if environment variable not set
            self.skipTest(f"Set {NONSIGMF_ENV} environment variable to path with BLUE files to run test.")

        # glob all files in blue/ directory
        blue_dir = NONSIGMF_REPO / "blue"
        self.blue_paths = []
        if blue_dir.exists():
            for ext in ["*.cdif", "*.tmp"]:
                self.blue_paths.extend(blue_dir.glob(f"**/{ext}"))
        if not self.blue_paths:
            self.fail(f"No BLUE files (*.cdif, *.tmp) found in {blue_dir}.")

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_sigmf_pair(self):
        """test standard blue to sigmf conversion with file pairs"""
        for blue_path in self.blue_paths:
            sigmf_path = self.tmp_path / blue_path.stem
            meta = blue_to_sigmf(blue_path=blue_path, out_path=sigmf_path)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            # FIXME: REPLACE BELOW WITH BELOW COMMENTED AFTER PR #121 MERGED
            if not meta.get_global_field("core:metadata_only"):
                _ = meta.read_samples(count=10)
                # check sample read consistency
                # np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_sigmf_archive(self):
        """test blue to sigmf conversion with archive output"""
        for blue_path in self.blue_paths:
            sigmf_path = self.tmp_path / f"{blue_path.stem}_archive"
            meta = blue_to_sigmf(blue_path=blue_path, out_path=sigmf_path, create_archive=True)
            # now read newly created archive
            arc_meta = sigmf.fromfile(sigmf_path)
            self.assertIsInstance(arc_meta, sigmf.SigMFFile)
            # FIXME: REPLACE BELOW WITH BELOW COMMENTED AFTER PR #121 MERGED
            if not arc_meta.get_global_field("core:metadata_only"):
                _ = arc_meta.read_samples(count=10)
                # check sample read consistency
                # np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_create_ncd(self):
        """test direct NCD conversion"""
        for blue_path in self.blue_paths:
            meta = blue_to_sigmf(blue_path=blue_path)
            _validate_ncd(self, meta, blue_path)

            # test that data can be read if not metadata-only
            if not meta.get_global_field("core:metadata_only"):
                _ = meta.read_samples(count=10)

    def test_autodetect_ncd(self):
        """test automatic NCD conversion"""
        for blue_path in self.blue_paths:
            meta = sigmf.fromfile(blue_path)
            _validate_ncd(self, meta, blue_path)

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Signal Hound Converter"""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import sigmf
from sigmf.convert.signalhound import signalhound_to_sigmf

from .test_convert_wav import _validate_ncd
from .testdata import get_nonsigmf_path


class TestSignalHoundWithNonSigMFRepo(unittest.TestCase):
    """Test Signal Hound converter with real example files if available."""

    def setUp(self) -> None:
        """Find a non-SigMF dataset for testing."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        nonsigmf_path = get_nonsigmf_path(self)
        # glob all files in signal hound directory
        hound_dir = nonsigmf_path / "signal_hound"
        self.hound_paths = []
        self.hound_paths.extend(hound_dir.glob("*.xml"))
        if not self.hound_paths:
            self.fail(f"No Signal Hound XML files found in {hound_dir}.")

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()

    def test_sigmf_pair(self):
        """test basic signal hound to sigmf conversion with file pairs"""
        for hound_path in self.hound_paths:
            sigmf_path = self.tmp_path / hound_path.stem
            meta = signalhound_to_sigmf(signalhound_path=hound_path, out_path=sigmf_path)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if not meta.get_global_field("core:metadata_only"):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_sigmf_archive(self):
        """test signal hound to sigmf conversion with archive output"""
        for hound_path in self.hound_paths:
            sigmf_path = self.tmp_path / f"{hound_path.stem}_archive"
            meta = signalhound_to_sigmf(signalhound_path=hound_path, out_path=sigmf_path, create_archive=True)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if not meta.get_global_field("core:metadata_only"):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_create_ncd(self):
        """test direct NCD conversion"""
        for hound_path in self.hound_paths:
            meta = signalhound_to_sigmf(signalhound_path=hound_path)
            _validate_ncd(self, meta, hound_path)
            if len(meta):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_fromfile_ncd(self):
        """test automatic NCD conversion with fromfile"""
        for hound_path in self.hound_paths:
            meta = sigmf.fromfile(hound_path)
            _validate_ncd(self, meta, hound_path)

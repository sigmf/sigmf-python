# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Converters"""

import unittest
import os
import tempfile
from pathlib import Path
import numpy as np

try:
    from scipy.io import wavfile

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import sigmf
from sigmf.apps.convert_wav import convert_wav


@unittest.skipUnless(SCIPY_AVAILABLE, "scipy is required for WAV file tests")
class TestWAVConverter(unittest.TestCase):
    def setUp(self) -> None:
        """create temp wav file for testing"""
        if not SCIPY_AVAILABLE:
            self.skipTest("scipy is required for WAV file tests")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.wav_path = self.tmp_path / "foo.wav"
        samp_rate = 48000
        duration_s = 0.1
        ttt = np.linspace(0, duration_s, int(samp_rate * duration_s), endpoint=False)
        freq = 440  # A4 note
        self.audio_data = 0.5 * np.sin(2 * np.pi * freq * ttt)
        wavfile.write(self.wav_path, samp_rate, self.audio_data.astype(np.float32))

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_wav_to_sigmf(self):
        sigmf_path = convert_wav(wav_path=self.wav_path, out_path=str(self.tmp_path / "bar"))
        meta = sigmf.fromfile(sigmf_path)
        data = meta.read_samples()
        # allow small numerical differences due to data type conversions
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-8))


class TestBlueConverter(unittest.TestCase):
    def setUp(self) -> None:
        # skip test if environment variable not set
        if not os.getenv("NONSIGMF_RECORDINGS_PATH"):
            self.skipTest("NONSIGMF_RECORDINGS_PATH environment variable needed for Bluefile tests.")

    def test_blue_to_sigmffile(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)

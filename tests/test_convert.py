# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Converters"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from scipy.io import wavfile

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import sigmf
from sigmf.apps.convert_blue import convert_blue
from sigmf.apps.convert_wav import convert_wav

BLUE_ENV_VAR = "NONSIGMF_RECORDINGS_PATH"


class TestWAVConverter(unittest.TestCase):
    """wav loopback test"""

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
        sigmf_path = self.tmp_path / "bar"
        _ = convert_wav(wav_path=self.wav_path, out_path=sigmf_path)
        meta = sigmf.fromfile(sigmf_path)
        data = meta.read_samples()
        # allow small numerical differences due to data type conversions
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-8))


class TestBlueConverter(unittest.TestCase):
    """As we have no blue files in the repository, test only when env path specified."""

    def setUp(self) -> None:
        blue_path = Path(os.getenv(BLUE_ENV_VAR, "nopath"))
        if not blue_path or blue_path == Path("nopath"):
            # skip test if environment variable not set
            self.skipTest(f"Set {BLUE_ENV_VAR} environment variable to location with .cdif files to run test.")
        if not blue_path.is_dir():
            self.fail(f"{blue_path} is not a valid directory.")
        self.bluefiles = list(blue_path.glob("*.cdif"))
        if not self.bluefiles:
            self.fail(f"No .cdif files found in {BLUE_ENV_VAR}.")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_blue_to_sigmf(self):
        for bdx, bluefile in enumerate(self.bluefiles):
            sigmf_path = self.tmp_path / f"converted_{bdx}"
            _ = convert_blue(blue_path=bluefile, out_path=sigmf_path)
            meta = sigmf.fromfile(sigmf_path)

            ### EVERYTHING BELOW HERE IS FOR DEBUGGING ONLY _ REMOVE LATER ###
            # plot stft of RF data for visual inspection
            from scipy.signal import spectrogram

            samples = meta.read_samples()
            freqs, times, spec = spectrogram(samples, fs=meta.get_global_field("core:sample_rate"), nperseg=1024)
            # use imshow to plot spectrogram
            import matplotlib.pyplot as plt

            plt.figure()
            plt.imshow(
                10 * np.log10(spec), aspect="auto", extent=[times[0], times[-1], freqs[0], freqs[-1]], origin="lower"
            )
            plt.colorbar(label="Intensity [dB]")
            plt.ylabel("Frequency [Hz]")
            plt.xlabel("Time [s]")
            plt.title(f"Spectrogram of {bluefile.name}")
            plt.show()
            self.assertIsInstance(meta, sigmf.SigMFFile)

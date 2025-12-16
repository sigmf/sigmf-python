# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Converters"""

import os
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import sigmf
from sigmf.convert.blue import blue_to_sigmf
from sigmf.convert.wav import wav_to_sigmf

BLUE_ENV_VAR = "NONSIGMF_RECORDINGS_PATH"


class TestWAVConverter(unittest.TestCase):
    """wav loopback test"""

    def setUp(self) -> None:
        """create temp wav file for testing"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.wav_path = self.tmp_path / "foo.wav"
        samp_rate = 48000
        duration_s = 0.1
        ttt = np.linspace(0, duration_s, int(samp_rate * duration_s), endpoint=False)
        freq = 440  # A4 note
        self.audio_data = 0.5 * np.sin(2 * np.pi * freq * ttt)
        # note scipy could write float wav files directly,
        # but to avoid adding scipy as a dependency for sigmf-python,
        # convert float audio to 16-bit PCM integer format
        audio_int16 = (self.audio_data * 32767).astype(np.int16)

        # write wav file using built-in wave module
        with wave.open(str(self.wav_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit = 2 bytes
            wav_file.setframerate(samp_rate)
            wav_file.writeframes(audio_int16.tobytes())

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_wav_to_sigmf(self):
        sigmf_path = self.tmp_path / "bar"
        meta = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path)
        data = meta.read_samples()
        # allow numerical differences due to PCM quantization
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))


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
            sigmf_path = self.tmp_path / bluefile.stem
            meta = blue_to_sigmf(blue_path=bluefile, out_path=sigmf_path)

            ### EVERYTHING BELOW HERE IS FOR DEBUGGING ONLY _ REMOVE LATER ###
            # plot stft of RF data for visual inspection
            import matplotlib.pyplot as plt
            from scipy.signal import spectrogram
            from swiftfox import summary

            samples = meta.read_samples()
            plt.figure(figsize=(10, 10))
            summary(samples, detail=0.1, samp_rate=meta.get_global_field("core:sample_rate"))
            plt.figure()
            plt.plot(samples.real)
            plt.plot(samples.imag)

            freqs, times, spec = spectrogram(samples, fs=meta.get_global_field("core:sample_rate"), nperseg=1024)
            # use imshow to plot spectrogram

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

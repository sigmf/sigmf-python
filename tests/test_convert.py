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

from .testdata import NONSIGMF_REPO, NONSIGMF_ENV


class TestWAVConverter(unittest.TestCase):
    """wav loopback test"""

    def setUp(self) -> None:
        """temp wav file for testing"""
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

    def test_wav_to_sigmf_pair(self):
        sigmf_path = self.tmp_path / "bar.tmp"
        meta = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path)
        data = meta.read_samples()
        # allow numerical differences due to PCM quantization
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["data_fn"].exists(), "dataset path missing")
        self.assertTrue(filenames["meta_fn"].exists(), "metadata path missing")

    def test_wav_to_sigmf_archive(self):
        sigmf_path = self.tmp_path / "baz.ext"
        wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_archive=True)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["archive_fn"].exists(), "archive path missing")


class TestBlueConverter(unittest.TestCase):
    """As we have no blue files in the repository, test only when env path specified."""

    def setUp(self) -> None:
        """temp paths & blue files"""
        if not NONSIGMF_REPO:
            # skip test if environment variable not set
            self.skipTest(f"Set {NONSIGMF_ENV} environment variable to path with .cdif files to run test. ")
        self.bluefiles = list(NONSIGMF_REPO.glob("**/*.cdif"))
        print("bluefiles", self.bluefiles)
        if not self.bluefiles:
            self.fail(f"No .cdif files found in {NONSIGMF_ENV}.")
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_blue_to_sigmf_pair(self):
        for bdx, bluefile in enumerate(self.bluefiles):
            sigmf_path = self.tmp_path / bluefile.stem
            meta = blue_to_sigmf(blue_path=bluefile, out_path=sigmf_path)
            print(f"Converted {bluefile} to SigMF at {sigmf_path}")
            if not meta.get_global_field("core:metadata_only"):
                print(meta.read_samples(count=10))

            # ### EVERYTHING BELOW HERE IS FOR DEBUGGING ONLY _ REMOVE LATER ###
            # # plot stft of RF data for visual inspection
            # import matplotlib.pyplot as plt
            # from scipy.signal import spectrogram
            # from swiftfox import summary, smartspec

            # if meta.get_global_field("core:metadata_only"):
            #     print("Metadata only file, skipping plot.")
            #     continue
            # samples = meta.read_samples()
            # # plt.figure(figsize=(10, 10))
            # summary(samples, detail=0.1, samp_rate=meta.get_global_field("core:sample_rate"), title=sigmf_path.name)
            # plt.figure()
            # # plt.plot(samples.real)
            # # plt.plot(samples.imag)
            # # plt.figure()
            # spec = smartspec(samples, detail=0.5, samp_rate=meta.get_global_field("core:sample_rate"))
            # # use imshow to plot spectrogram

            # plt.show()
            self.assertIsInstance(meta, sigmf.SigMFFile)

    def test_blue_to_sigmf_archive(self):
        for bdx, bluefile in enumerate(self.bluefiles):
            sigmf_path = self.tmp_path / f"{bluefile.stem}_archive"
            meta = blue_to_sigmf(blue_path=bluefile, out_path=sigmf_path, create_archive=True)
            print(f"Converted {bluefile} to SigMF archive at {sigmf_path}")
            self.assertIsInstance(meta, sigmf.SigMFFile)

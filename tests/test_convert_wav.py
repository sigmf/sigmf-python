# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for WAV Converter"""

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

import sigmf
from sigmf.convert.wav import wav_to_sigmf

from .testdata import get_nonsigmf_path


def _validate_ncd(test: unittest.TestCase, meta: sigmf.SigMFFile, target_path: Path):
    """non-conforming dataset has a specific structure"""
    test.assertEqual(str(meta.data_file), str(target_path), "Auto-detected NCD should point to original file")
    test.assertIsInstance(meta, sigmf.SigMFFile)

    global_info = meta.get_global_info()
    capture_info = meta.get_captures()

    # validate NCD SigMF spec compliance
    test.assertGreater(len(capture_info), 0, "Should have at least one capture")
    test.assertIn("core:header_bytes", capture_info[0])
    test.assertGreater(capture_info[0]["core:header_bytes"], 0, "Should have non-zero core:header_bytes field")
    test.assertIn("core:trailing_bytes", global_info, "Should have core:trailing_bytes field.")
    test.assertIn("core:dataset", global_info, "Should have core:dataset field.")
    test.assertNotIn("core:metadata_only", global_info, "Should NOT have core:metadata_only field.")


class TestWAVConverter(unittest.TestCase):
    """Create a realistic WAV file and test conversion methods."""

    def setUp(self) -> None:
        """temp WAV file with tone for testing"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.wav_path = self.tmp_path / "foo.wav"
        samp_rate = 48000
        duration_s = 0.1
        ttt = np.linspace(0, duration_s, int(samp_rate * duration_s), endpoint=False)
        freq = 440  # A4 note
        self.audio_data = 0.5 * np.sin(2 * np.pi * freq * ttt)
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

    def test_wav_to_sigmf_pair(self) -> None:
        """test standard wav to sigmf conversion with file pairs"""
        sigmf_path = self.tmp_path / "bar"
        meta = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["data_fn"].exists(), "dataset path missing")
        self.assertTrue(filenames["meta_fn"].exists(), "metadata path missing")
        # verify data
        data = meta.read_samples()
        self.assertGreater(len(data), 0, "Should read some samples")
        # allow numerical differences due to PCM quantization
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))

        # test overwrite protection
        with self.assertRaises(sigmf.error.SigMFFileError) as context:
            wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, overwrite=False)
        self.assertIn("already exists", str(context.exception))

        # test overwrite works
        meta2 = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, overwrite=True)
        self.assertIsInstance(meta2, sigmf.SigMFFile)

    def test_wav_to_sigmf_archive(self) -> None:
        """test wav to sigmf conversion with archive output"""
        sigmf_path = self.tmp_path / "baz.ext"
        meta = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_archive=True)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["archive_fn"].exists(), "archive path missing")
        # verify data
        data = meta.read_samples()
        self.assertGreater(len(data), 0, "Should read some samples")
        # allow numerical differences due to PCM quantization
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))

        # test overwrite protection
        with self.assertRaises(sigmf.error.SigMFFileError) as context:
            wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_archive=True, overwrite=False)
        self.assertIn("already exists", str(context.exception))

        # test overwrite works
        meta2 = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_archive=True, overwrite=True)
        self.assertIsInstance(meta2, sigmf.SigMFFile)

    def test_wav_to_sigmf_ncd(self) -> None:
        """test wav to sigmf conversion as Non-Conforming Dataset"""
        meta = wav_to_sigmf(wav_path=self.wav_path, create_ncd=True)
        _validate_ncd(self, meta, self.wav_path)

        # verify data
        data = meta.read_samples()
        # allow numerical differences due to PCM quantization
        self.assertGreater(len(data), 0, "Should read some samples")
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))

        # test overwrite protection when creating NCD with output path
        sigmf_path = self.tmp_path / "ncd_test"
        wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_ncd=True, overwrite=True)
        with self.assertRaises(sigmf.error.SigMFFileError) as context:
            wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_ncd=True, overwrite=False)
        self.assertIn("already exists", str(context.exception))

        # test overwrite works
        meta2 = wav_to_sigmf(wav_path=self.wav_path, out_path=sigmf_path, create_ncd=True, overwrite=True)
        self.assertIsInstance(meta2, sigmf.SigMFFile)


class TestWAVWithNonSigMFRepo(unittest.TestCase):
    """Test WAV converter with real example files if available"""

    def setUp(self) -> None:
        """setup paths to example wav files"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        nonsigmf_path = get_nonsigmf_path(self)
        # glob all files in wav/ directory
        wav_dir = nonsigmf_path / "wav"
        self.wav_paths = []
        if wav_dir.exists():
            self.wav_paths = list(wav_dir.glob("*.wav"))
        if not self.wav_paths:
            self.fail(f"No WAV files (*.wav) found in {wav_dir} directory.")

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_sigmf_pair(self) -> None:
        """test standard wav to sigmf conversion with file pairs"""
        for wav_path in self.wav_paths:
            sigmf_path = self.tmp_path / wav_path.stem
            meta = wav_to_sigmf(wav_path=wav_path, out_path=sigmf_path)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if not meta.get_global_field("core:metadata_only"):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_sigmf_archive(self) -> None:
        """test wav to sigmf conversion with archive output"""
        for wav_path in self.wav_paths:
            sigmf_path = self.tmp_path / f"{wav_path.stem}_archive"
            meta = wav_to_sigmf(wav_path=wav_path, out_path=sigmf_path, create_archive=True)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if not meta.get_global_field("core:metadata_only"):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_create_ncd(self) -> None:
        """test direct NCD conversion"""
        for wav_path in self.wav_paths:
            meta = wav_to_sigmf(wav_path=wav_path)
            _validate_ncd(self, meta, wav_path)

            # test file read
            _ = meta.read_samples(count=10)

    def test_autodetect_ncd(self) -> None:
        """test automatic NCD conversion"""
        for wav_path in self.wav_paths:
            meta = sigmf.fromfile(wav_path)
            _validate_ncd(self, meta, wav_path)

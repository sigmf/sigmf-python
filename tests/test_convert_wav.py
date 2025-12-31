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
from typing import cast

import numpy as np

import sigmf
from sigmf.convert.wav import wav_to_sigmf

from .testdata import NONSIGMF_ENV, NONSIGMF_REPO


class TestWAVConverter(unittest.TestCase):
    """wav converter tests"""

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

    def _validate_dataset_key(self, meta, expected_filename):
        """validate DATASET_KEY is correctly set"""
        dataset_filename = meta.get_global_field("core:dataset")
        self.assertEqual(dataset_filename, expected_filename, "DATASET_KEY should contain filename")
        self.assertIsInstance(dataset_filename, str, "DATASET_KEY should be a string")

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

    def test_wav_to_sigmf_pair(self):
        """test standard wav to sigmf conversion with file pairs"""
        sigmf_path = self.tmp_path / "bar.tmp"
        meta = wav_to_sigmf(wav_path=str(self.wav_path), out_path=str(sigmf_path))
        data = meta.read_samples()
        # allow numerical differences due to PCM quantization
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))
        self.assertGreater(len(data), 0, "Should read some samples")
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["data_fn"].exists(), "dataset path missing")
        self.assertTrue(filenames["meta_fn"].exists(), "metadata path missing")

    def test_wav_to_sigmf_archive(self):
        """test wav to sigmf conversion with archive output"""
        sigmf_path = self.tmp_path / "baz.ext"
        wav_to_sigmf(wav_path=str(self.wav_path), out_path=str(sigmf_path), create_archive=True)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["archive_fn"].exists(), "archive path missing")

    def test_wav_to_sigmf_ncd(self):
        """test wav to sigmf conversion as Non-Conforming Dataset"""
        meta = wav_to_sigmf(wav_path=str(self.wav_path), create_ncd=True)

        # validate basic NCD structure
        captures = self._validate_ncd_structure(meta, self.wav_path)
        self.assertEqual(len(captures), 1, "Should have exactly one capture")

        # validate DATASET_KEY is set for NCD
        self._validate_dataset_key(meta, self.wav_path.name)

        # header_bytes should be non-zero for WAV files
        header_bytes = captures[0]["core:header_bytes"]
        self.assertGreater(header_bytes, 0, "WAV files should have non-zero header bytes")

        # verify data can still be read correctly from NCD
        data = meta.read_samples()
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))
        self.assertGreater(len(data), 0, "Should read some samples")

        # verify this is metadata-only (no separate data file created)
        self.assertIsInstance(meta.data_buffer, type(meta.data_buffer))

    def test_wav_auto_detection(self):
        """test automatic WAV detection through fromfile()"""
        # validate auto-detection works
        meta_raw = self._validate_auto_detection(self.wav_path)
        meta = cast(sigmf.SigMFFile, meta_raw)

        # validate DATASET_KEY is set for auto-detected NCD
        self._validate_dataset_key(meta, self.wav_path.name)

        # verify data can be read correctly
        data = meta.read_samples()
        self.assertTrue(np.allclose(self.audio_data, data, atol=1e-4))
        self.assertGreater(len(data), 0, "Should read some samples")


class TestWAVConverterWithRealFiles(unittest.TestCase):
    """Test WAV converter with real example files if available"""

    def setUp(self) -> None:
        """setup paths to example wav files"""
        self.wav_dir = None
        if NONSIGMF_REPO:
            wav_path = NONSIGMF_REPO / "wav"
            if wav_path.exists():
                self.wav_dir = wav_path
                self.wav_files = list(wav_path.glob("*.wav"))

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

    def _validate_dataset_key(self, meta, expected_filename):
        """validate DATASET_KEY is correctly set"""
        dataset_filename = meta.get_global_field("core:dataset")
        self.assertEqual(dataset_filename, expected_filename, "DATASET_KEY should contain filename")

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

    def test_real_wav_files_ncd(self):
        """test NCD conversion with real example wav files"""
        if not self.wav_dir or not hasattr(self, "wav_files"):
            self.skipTest(f"Set {NONSIGMF_ENV} environment variable to path with wav/ directory to run test.")

        if not self.wav_files:
            self.skipTest(f"No .wav files found in {self.wav_dir}")

        for wav_file in self.wav_files:
            meta = wav_to_sigmf(wav_path=str(wav_file), create_ncd=True)

            # validate basic NCD structure
            self._validate_ncd_structure(meta, wav_file)

            # validate auto-detection also works
            meta_auto = self._validate_auto_detection(wav_file)
            self._validate_dataset_key(meta_auto, wav_file.name)

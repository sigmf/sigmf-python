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

from .testdata import get_nonsigmf_path, validate_ncd


class TestSignalHoundConverter(unittest.TestCase):
    """Create a realistic Signal Hound XML/IQ file pair and test conversion methods."""

    def setUp(self) -> None:
        """Create temp XML/IQ file pair with tone for testing."""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.iq_path = self.tmp_path / "test.iq"
        self.xml_path = self.tmp_path / "test.xml"

        # Generate complex IQ test data
        self.samp_rate = 48000
        self.center_freq = 915e6
        duration_s = 0.1
        num_samples = int(self.samp_rate * duration_s)
        ttt = np.linspace(0, duration_s, num_samples, endpoint=False)
        freq = 440  # A4 note
        self.iq_data = 0.5 * np.exp(2j * np.pi * freq * ttt)  # complex128, normalized to [-0.5, 0.5]

        # Convert complex IQ data to interleaved int16 format (ci16_le - Signal Hound "Complex Short")
        scale = 2**15  # int16 range is -32768 to 32767
        ci_real = (self.iq_data.real * scale).astype(np.int16)
        ci_imag = (self.iq_data.imag * scale).astype(np.int16)
        iq_interleaved = np.empty((len(self.iq_data) * 2,), dtype=np.int16)
        iq_interleaved[0::2] = ci_real
        iq_interleaved[1::2] = ci_imag

        # Write IQ file as raw interleaved int16
        with open(self.iq_path, "wb") as iq_file:
            iq_file.write(iq_interleaved.tobytes())

        # Write minimal XML metadata file
        with open(self.xml_path, "w") as xml_file:
            xml_file.write(
                f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<SignalHoundIQFile Version="1.0">\n'
                f"    <CenterFrequency>{self.center_freq}</CenterFrequency>\n"
                f"    <SampleRate>{self.samp_rate}</SampleRate>\n"
                f"    <DataType>Complex Short</DataType>\n"
                f"    <IQFileName>{self.iq_path.name}</IQFileName>\n"
                f"</SignalHoundIQFile>\n"
            )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.tmp_dir.cleanup()

    def _verify(self, meta: sigmf.SigMFFile) -> None:
        """Verify metadata fields and data integrity."""
        self.assertIsInstance(meta, sigmf.SigMFFile)
        self.assertEqual(meta.get_global_field("core:datatype"), "ci16_le")
        self.assertEqual(meta.get_global_field("core:sample_rate"), self.samp_rate)
        # center frequency is in capture metadata
        self.assertEqual(meta.get_captures()[0]["core:frequency"], self.center_freq)
        # verify data
        data = meta.read_samples()
        self.assertGreater(len(data), 0, "Should read some samples")
        # allow numerical differences due to int16 quantization
        self.assertTrue(np.allclose(self.iq_data, data, atol=1e-4))

    def test_signalhound_to_sigmf_pair(self):
        """Test standard Signal Hound to SigMF conversion with file pairs."""
        sigmf_path = self.tmp_path / "converted"
        meta = signalhound_to_sigmf(signalhound_path=self.xml_path, out_path=sigmf_path)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["data_fn"].exists(), "dataset path missing")
        self.assertTrue(filenames["meta_fn"].exists(), "metadata path missing")
        self._verify(meta)

        # test overwrite protection
        with self.assertRaises(sigmf.error.SigMFFileError) as context:
            signalhound_to_sigmf(signalhound_path=self.xml_path, out_path=sigmf_path, overwrite=False)
        self.assertIn("already exists", str(context.exception))

        # test overwrite works
        meta2 = signalhound_to_sigmf(signalhound_path=self.xml_path, out_path=sigmf_path, overwrite=True)
        self.assertIsInstance(meta2, sigmf.SigMFFile)

    def test_signalhound_to_sigmf_archive(self):
        """Test Signal Hound to SigMF conversion with archive output."""
        sigmf_path = self.tmp_path / "converted_archive"
        meta = signalhound_to_sigmf(signalhound_path=self.xml_path, out_path=sigmf_path, create_archive=True)
        filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
        self.assertTrue(filenames["archive_fn"].exists(), "archive path missing")
        self._verify(meta)

        # test overwrite protection
        with self.assertRaises(sigmf.error.SigMFFileError) as context:
            signalhound_to_sigmf(
                signalhound_path=self.xml_path, out_path=sigmf_path, create_archive=True, overwrite=False
            )
        self.assertIn("already exists", str(context.exception))

    def test_signalhound_to_sigmf_ncd(self):
        """Test Signal Hound to SigMF conversion as Non-Conforming Dataset."""
        meta = signalhound_to_sigmf(signalhound_path=self.xml_path, create_ncd=True)
        target_path = self.iq_path
        validate_ncd(self, meta, target_path)
        self._verify(meta)


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
            target_path = hound_path.with_suffix(".iq")
            validate_ncd(self, meta, target_path)
            if len(meta):
                # check sample read consistency
                np.testing.assert_array_equal(meta.read_samples(count=10), meta[0:10])

    def test_fromfile_ncd(self):
        """test automatic NCD conversion with fromfile"""
        for hound_path in self.hound_paths:
            meta = sigmf.fromfile(hound_path)
            target_path = hound_path.with_suffix(".iq")
            validate_ncd(self, meta, target_path)

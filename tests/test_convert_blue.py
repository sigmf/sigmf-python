# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for BLUE Converter"""

import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sigmf
from sigmf.convert.blue import TYPE_MAP, blue_to_sigmf
from sigmf.utils import SIGMF_DATETIME_ISO8601_FMT

from .test_convert_wav import _validate_ncd
from .testdata import get_nonsigmf_path


class TestBlueConverter(unittest.TestCase):
    """Create minimal BLUE file and test conversion methods."""

    def setUp(self) -> None:
        """temp BLUE file with minimal data for testing"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        self.blue_path = self.tmp_path / "foo.cdif"
        self.format_tolerance = [
            ("SB", 1e-1),  # scalar int8
            ("CB", 1e-1),  # complex int8
            ("SU", 1e-4),  # scalar uint16
            ("CU", 1e-4),  # complex uint16
            ("SI", 1e-4),  # scalar int16
            ("CI", 1e-4),  # complex int16
            ("SV", 1e-7),  # scalar uint32
            ("CV", 1e-7),  # complex uint32
            ("SL", 1e-8),  # scalar int32
            ("CL", 1e-8),  # complex int32
            # ("SX", 1e-8),  # scalar int64, should work but not allowed by SigMF spec
            # ("CX", 1e-8),  # complex int64, should work but not allowed by SigMF spec
            ("SF", 1e-8),  # scalar float32
            ("CF", 1e-8),  # complex float32
            ("SD", 0),  # scalar float64
            ("CD", 0),  # complex float64
        ]

        self.samp_rate = 192e3
        num_samples = 1024
        ttt = np.linspace(0, num_samples / self.samp_rate, num_samples, endpoint=False)
        freq = 3520  # A7 note
        self.iq_data = 0.5 * np.exp(2j * np.pi * freq * ttt)  # complex128
        time_now = datetime.now(timezone.utc)
        self.datetime = time_now.strftime(SIGMF_DATETIME_ISO8601_FMT)
        self.timecode = (time_now - datetime(1950, 1, 1, tzinfo=timezone.utc)).total_seconds()

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def write_minimal(self, format: bytes = b"CF") -> None:
        """write minimal BLUE file to self.blue_path"""

        is_complex = format[0] == ord(b"C")
        dtype = TYPE_MAP[chr(format[1])]

        if np.issubdtype(dtype, np.integer):
            scale = 2 ** (np.dtype(dtype).itemsize * 8 - 1)
            if is_complex:
                if np.dtype(dtype).kind == "u":
                    # unsigned
                    ci_real = (self.iq_data.real * scale + scale).astype(dtype)
                    ci_imag = (self.iq_data.imag * scale + scale).astype(dtype)
                else:
                    # signed
                    ci_real = (self.iq_data.real * scale).astype(dtype)
                    ci_imag = (self.iq_data.imag * scale).astype(dtype)
                iq_converted = np.empty((self.iq_data.size * 2,), dtype=dtype)
                iq_converted[0::2] = ci_real
                iq_converted[1::2] = ci_imag
            else:
                if np.dtype(dtype).kind == "u":
                    # unsigned
                    iq_converted = (self.iq_data.real * scale + scale).astype(dtype)
                else:
                    # signed
                    iq_converted = (self.iq_data.real * scale).astype(dtype)
        elif np.issubdtype(dtype, np.floating):
            if is_complex:
                ci_real = self.iq_data.real.astype(dtype)
                ci_imag = self.iq_data.imag.astype(dtype)
                iq_converted = np.empty((self.iq_data.size * 2,), dtype=dtype)
                iq_converted[0::2] = ci_real
                iq_converted[1::2] = ci_imag
            else:
                iq_converted = self.iq_data.real.astype(dtype)
        else:
            raise ValueError(f"unsupported dtype for BLUE conversion: {dtype}")

        with open(self.blue_path, "wb") as handle:
            # empty hcb
            handle.write(b"\x00" * 512)
            # minimal fields
            handle.seek(0)
            handle.write(b"BLUEEEEIEEEI")  # magic & endianness
            handle.seek(32)
            handle.write(
                struct.pack("<ddi2s", 512, iq_converted.nbytes, 1000, format)
            )  # data_start, data_size, type, format
            handle.seek(56)
            handle.write(struct.pack("<d", self.timecode))  # timecode
            handle.seek(256)
            handle.write(struct.pack("<ddi", 0, 1 / self.samp_rate, 0))  # xstart, xdelta, xunits
            handle.seek(512)
            handle.write(iq_converted.tobytes())  # write IQ data

    def check_data_and_metadata(self, meta, form, atol):
        """ensure blue is sample perfect and metadata is equivalent"""
        self.assertEqual(meta.sample_rate, self.samp_rate)
        self.assertEqual(meta.get_capture_info(0)["core:datetime"], self.datetime)
        if form[0] == "S":
            np.testing.assert_allclose(self.iq_data.real, meta[:], atol=atol)
        else:
            np.testing.assert_allclose(self.iq_data, meta[:], atol=atol)

    def test_blue_to_sigmf_pair(self) -> None:
        """test standard blue to sigmf conversion with file pairs"""
        for form, atol in self.format_tolerance:
            sigmf_path = self.tmp_path / f"bar{format}"
            self.write_minimal(form.encode())
            meta = blue_to_sigmf(blue_path=self.blue_path, out_path=sigmf_path)
            filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
            self.assertTrue(filenames["data_fn"].exists(), "dataset path missing")
            self.assertTrue(filenames["meta_fn"].exists(), "metadata path missing")
            self.check_data_and_metadata(meta, form, atol)

    def test_blue_to_sigmf_archive(self) -> None:
        """test blue to sigmf conversion with archive output"""
        for form, atol in self.format_tolerance:
            self.write_minimal(form.encode())
            sigmf_path = self.tmp_path / f"baz{format}"
            meta = blue_to_sigmf(blue_path=self.blue_path, out_path=sigmf_path, create_archive=True)
            filenames = sigmf.sigmffile.get_sigmf_filenames(sigmf_path)
            self.assertTrue(filenames["archive_fn"].exists(), "archive path missing")
            self.check_data_and_metadata(meta, form, atol)

    def test_blue_to_sigmf_ncd(self) -> None:
        """test automatic NCD conversion with fromfile()"""
        for form, atol in self.format_tolerance:
            self.write_minimal(form.encode())
            meta = blue_to_sigmf(self.blue_path)
            _validate_ncd(self, meta, self.blue_path)
            self.check_data_and_metadata(meta, form, atol)


class TestBlueWithNonSigMFRepo(unittest.TestCase):
    """BLUE converter tests using external files"""

    def setUp(self) -> None:
        """setup paths to blue files"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        nonsigmf_path = get_nonsigmf_path(self)
        # glob all files in blue/ directory
        blue_dir = nonsigmf_path / "blue"
        self.blue_paths = []
        if blue_dir.exists():
            for ext in ["*.cdif", "*.tmp"]:
                self.blue_paths.extend(blue_dir.glob(f"**/{ext}"))
        if not self.blue_paths:
            self.fail(f"No BLUE files (*.cdif, *.tmp) found in {blue_dir} directory.")

    def tearDown(self) -> None:
        """clean up temporary directory"""
        self.tmp_dir.cleanup()

    def test_sigmf_pair(self):
        """test standard blue to sigmf conversion with file pairs"""
        for blue_path in self.blue_paths:
            sigmf_path = self.tmp_path / blue_path.stem
            meta = blue_to_sigmf(blue_path=blue_path, out_path=sigmf_path)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if not meta.get_global_field("core:metadata_only"):
                # check sample read consistency
                np.testing.assert_allclose(meta.read_samples(count=10), meta[0:10], atol=1e-6)

    def test_sigmf_archive(self):
        """test blue to sigmf conversion with archive output"""
        for blue_path in self.blue_paths:
            sigmf_path = self.tmp_path / f"{blue_path.stem}_archive"
            meta = blue_to_sigmf(blue_path=blue_path, out_path=sigmf_path, create_archive=True)
            self.assertIsInstance(meta, sigmf.SigMFFile)
            if len(meta):
                # check sample read consistency
                np.testing.assert_allclose(meta.read_samples(count=10), meta[0:10], atol=1e-6)

    def test_create_ncd(self):
        """test direct NCD conversion"""
        for blue_path in self.blue_paths:
            meta = blue_to_sigmf(blue_path=blue_path)
            _validate_ncd(self, meta, blue_path)
            if len(meta):
                # check sample read consistency
                np.testing.assert_allclose(meta.read_samples(count=10), meta[0:10], atol=1e-6)

    def test_fromfile_ncd(self):
        """test automatic NCD conversion with fromfile()"""
        for blue_path in self.blue_paths:
            meta = sigmf.fromfile(blue_path)
            _validate_ncd(self, meta, blue_path)

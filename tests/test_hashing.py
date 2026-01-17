# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Hashing Tests"""

import io
import shutil
import tempfile
import unittest
from copy import deepcopy
from hashlib import sha512
from pathlib import Path

import numpy as np

from sigmf import SigMFFile, hashing

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


class TestHashCalculation(unittest.TestCase):
    """Test hash calculation consistency across different SigMF formats."""

    def setUp(self):
        """Set up temporary directory for tests."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_ncd_hash_covers_entire_file(self):
        """Test that non-conforming datasets hash the entire file including headers."""
        data_path = self.temp_dir / "ncd.bin"
        with open(data_path, "wb") as handle:
            # Create NCD file with header, data, and trailer
            handle.write(b"\x00" * 64)  # header
            handle.write(TEST_FLOAT32_DATA.tobytes())  # sample data
            handle.write(b"\xFF" * 32)  # trailer

        # Create SigMF metadata for NCD
        ncd_metadata = deepcopy(TEST_METADATA)
        del ncd_metadata["global"][SigMFFile.HASH_KEY]
        ncd_metadata["global"][SigMFFile.TRAILING_BYTES_KEY] = 32
        meta = SigMFFile(metadata=ncd_metadata)
        meta.set_data_file(data_path, offset=64)

        file_hash = hashing.calculate_sha512(filename=data_path)
        sigmf_hash = meta.get_global_field(SigMFFile.HASH_KEY)
        self.assertEqual(file_hash, sigmf_hash)

    def test_edge_cases(self):
        """Test edge cases in hash calculation function."""
        # empty file
        empty_file = self.temp_dir / "empty.dat"
        empty_file.touch()
        empty_hash = hashing.calculate_sha512(filename=empty_file)
        empty_hash_expected = sha512(b"").hexdigest()
        self.assertEqual(empty_hash, empty_hash_expected)

        # small file (less than 4096 bytes)
        small_data = np.random.bytes(128)
        small_hash_expected = sha512(small_data).hexdigest()
        small_file = self.temp_dir / "small.dat"
        small_file.write_bytes(small_data)
        small_hash = hashing.calculate_sha512(filename=small_file)
        self.assertEqual(small_hash, small_hash_expected)

        # BytesIO
        buffer = io.BytesIO(small_data)
        buffer_hash = hashing.calculate_sha512(fileobj=buffer)
        self.assertEqual(buffer_hash, small_hash_expected)

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests wav formatted audio conversion"""

import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

from sigmf.apps.convert_wav import convert_wav


def test_wav_to_sigmf_basic():
    """Basic smoke-test: convert a tiny WAV → SIGMF, assert file created."""
    fs = 48_000
    t = np.linspace(0, 0.1, int(fs * 0.1))  # 0.1 s
    sine = np.sin(2 * np.pi * 1000 * t)
    sine_int = (sine * 32767).astype(np.int16)

    # Create temp file and close it before use
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name

    # Write to the closed file
    wavfile.write(tmp_wav_path, fs, sine_int)
    tmp_sigmf = tmp_wav_path.replace(".wav", ".sigmf")

    try:
        # Run converter
        convert_wav(tmp_wav_path, tmp_sigmf)

        # Assert SIGMF file exists and non-zero
        assert os.path.exists(tmp_sigmf), "SIGMF file not created"
        assert os.path.getsize(tmp_sigmf) > 0, "SIGMF file is empty"
    finally:
        # Clean up both files
        if os.path.exists(tmp_wav_path):
            os.remove(tmp_wav_path)
        if os.path.exists(tmp_sigmf):
            os.remove(tmp_sigmf)

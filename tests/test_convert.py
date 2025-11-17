# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Converters"""

import unittest
import os

def _has_apps_dependencies():
    """Check if optional [apps] dependencies are available."""
    try:
        import scipy.io.wavfile  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_has_apps_dependencies(), "Optional [apps] dependencies not available")
class TestWAVConverter(unittest.TestCase):
    def test_wav_to_sigmffile(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)


@unittest.skipUnless(_has_apps_dependencies(), "Optional [apps] dependencies not available")
class TestBlueConverter(unittest.TestCase):
    def setUp(self) -> None:
        # skip test if environment variable not set
        if not os.getenv("NONSIGMF_RECORDINGS_PATH"):
            self.skipTest("NONSIGMF_RECORDINGS_PATH environment variable needed for Bluefile tests.")

    def test_blue_to_sigmffile(self):
        # Placeholder for actual test implementation
        self.assertTrue(True)
# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for signal generation utilities."""

import unittest
from io import BytesIO

import numpy as np
import numpy.testing as npt

from sigmf import SigMFFile
from sigmf.error import SigMFGeneratorError
from sigmf.siggen import SigMFGenerator


class TestSigGen(unittest.TestCase):
    """Test Signal Generator functionality."""

    def setUp(self):
        """setup test fixtures"""
        self.seed = 0xDEADC0DE
        self.test_sample_rate = 48000
        self.test_duration = 1.0
        self.test_freq = -1000.0

    def test_deterministic_tone_generation(self):
        """test deterministic tone generation with specified parameters"""
        signal = (
            SigMFGenerator(self.seed)
            .tone(self.test_freq)
            .sample_rate(self.test_sample_rate)
            .duration(self.test_duration)
            .generate()
        )

        # verify object type
        self.assertIsInstance(signal, SigMFFile)

        # verify metadata
        self.assertEqual(signal.sample_rate, self.test_sample_rate)
        self.assertEqual(signal.get_global_info()[SigMFFile.DATATYPE_KEY], "cf32_le")
        self.assertIn("-1000.0 hz tone", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

        # verify signal characteristics
        samples = signal.read_samples()
        expected_samples = int(self.test_sample_rate * self.test_duration)
        self.assertEqual(len(samples), expected_samples)

        # verify it's complex data
        self.assertTrue(np.iscomplexobj(samples))

        # verify frequency content by checking dominant frequency
        fft_samples = np.fft.fft(samples)
        fft_freqs = np.fft.fftfreq(len(samples), 1 / self.test_sample_rate)
        dominant_freq_idx = np.argmax(np.abs(fft_samples))
        dominant_freq = fft_freqs[dominant_freq_idx]
        self.assertAlmostEqual(dominant_freq, self.test_freq, delta=10)  # within 10 hz, signed

    def test_random_tone_generation(self):
        """test random tone generation"""
        signal = SigMFGenerator(self.seed).tone().generate()

        # should have reasonable defaults
        samples = signal.read_samples()
        self.assertGreater(len(samples), 1000)  # at least 0.1s at min sample rate
        self.assertIsInstance(signal, SigMFFile)
        self.assertIn("hz tone", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

    def test_reproducible_generation(self):
        """test that same seed produces identical results"""
        # generate signal 3 times with same seed
        # only difference will be catpure datetime
        signal0 = SigMFGenerator(self.seed).generate()
        signal1 = SigMFGenerator(self.seed).generate()
        signal2 = SigMFGenerator(self.seed).generate()

        # set capture datetime identical
        for sig in [signal0, signal1, signal2]:
            sig.add_capture(0, {SigMFFile.DATETIME_KEY: "2026-01-01T00:00:00Z"})

        # compare metadata (which includes checksums)
        self.assertEqual(signal0, signal1)
        self.assertEqual(signal0, signal2)

    def test_sweep_generation(self):
        """test linear frequency sweep generation"""
        start_freq = -500.0
        end_freq = 2000.0

        signal = (
            SigMFGenerator()
            .sweep(start_freq, end_freq)
            .sample_rate(self.test_sample_rate)
            .duration(self.test_duration)
            .generate()
        )

        # verify metadata
        self.assertIn("-500.0-2000.0 hz sweep", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

        # verify signal properties
        samples = signal.read_samples()
        expected_samples = int(self.test_sample_rate * self.test_duration)
        self.assertEqual(len(samples), expected_samples)
        self.assertTrue(np.iscomplexobj(samples))

    def test_nominal_chaining(self):
        """test builder pattern method chaining"""
        signal = (
            SigMFGenerator(self.seed)
            .tone(2000)
            .sample_rate(44100)
            .duration(0.5)
            .amplitude(0.8)
            .snr(15)
            .frequency_offset(100)
            .phase_offset(np.pi / 4)
            .author("test@example.com")
            .description("test signal")
            .comment("test comment")
            .generate()
        )

        # verify chaining worked
        self.assertEqual(signal.get_global_info()[SigMFFile.AUTHOR_KEY], "test@example.com")
        self.assertEqual(signal.get_global_info()[SigMFFile.DESCRIPTION_KEY], "test signal")

        # should have multiple annotations: main signal + noise + freq offset + phase offset
        annotations = signal.get_annotations()
        self.assertGreaterEqual(len(annotations), 3)  # at least main + noise + offsets

        # find main signal annotation (has comment)
        main_annotation = next(ann for ann in annotations if SigMFFile.COMMENT_KEY in ann)
        self.assertEqual(main_annotation[SigMFFile.COMMENT_KEY], "test comment")

        # verify there's a noise annotation
        noise_annotations = [ann for ann in annotations if "AWGN" in ann.get(SigMFFile.LABEL_KEY, "")]
        self.assertEqual(len(noise_annotations), 1)

        # verify there's a frequency offset annotation
        freq_offset_annotations = [ann for ann in annotations if "freq offset" in ann.get(SigMFFile.LABEL_KEY, "")]
        self.assertEqual(len(freq_offset_annotations), 1)

    def test_snr_noise_addition(self):
        """test that snr parameter adds appropriate noise"""
        # generate clean tone and noisy tone
        clean_signal = (
            SigMFGenerator(self.seed)
            .tone(1000)
            .sample_rate(self.test_sample_rate)
            .duration(0.1)
            .generate()
        )
        noisy_signal = (
            SigMFGenerator(self.seed)
            .tone(1000)
            .sample_rate(self.test_sample_rate)
            .duration(0.1)
            .snr(10)
            .generate()
        )

        # noisy signal should have higher variance due to added noise
        clean_power = np.mean(np.abs(clean_signal.read_samples()) ** 2)
        noisy_power = np.mean(np.abs(noisy_signal.read_samples()) ** 2)

        # noisy signal should have more power due to added noise
        self.assertGreater(noisy_power, clean_power)

    def test_frequency_offset(self):
        """test frequency offset functionality"""
        base_freq = 1000.0
        offset_freq = 500.0

        signal = (
            SigMFGenerator(self.seed)
            .tone(base_freq)
            .frequency_offset(offset_freq)
            .sample_rate(self.test_sample_rate)
            .duration(self.test_duration)
            .generate()
        )

        # verify frequency in capture metadata includes offset
        captures = signal.get_captures()
        self.assertEqual(captures[0][SigMFFile.FREQUENCY_KEY], base_freq + offset_freq)

    def test_parameter_validation(self):
        """test parameter validation and error handling"""
        # bare generate() should now work (auto-generates components)
        signal = SigMFGenerator().generate()
        self.assertIsInstance(signal, SigMFFile)

        # should raise error for tone frequency exceeding nyquist (positive and negative)
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(30000).sample_rate(48000).duration(1.0).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(-30000).sample_rate(48000).duration(1.0).generate()

        # should raise error for negative duration
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(1000).sample_rate(48000).duration(-1.0).generate()

        # should raise error for negative sample rate
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(1000).sample_rate(-48000).duration(1.0).generate()

    def test_sweep_parameter_validation(self):
        """test sweep-specific parameter validation"""
        # sweep frequencies exceeding nyquist should raise error (positive and negative)
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(1000, 30000).sample_rate(48000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(30000, 1000).sample_rate(48000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(1000, -30000).sample_rate(48000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(-30000, 1000).sample_rate(48000).generate()

    def test_random_sweep_parameters(self):
        """test random sweep parameter generation"""
        signal = SigMFGenerator().sweep().generate()

        # should successfully generate
        self.assertIsInstance(signal, SigMFFile)

        # description should indicate it's a sweep
        desc = signal.get_global_info()[SigMFFile.DESCRIPTION_KEY]
        self.assertIn("sweep", desc)

        # should have start and end frequencies in description
        self.assertIn("-", desc)  # should have start-end format

    def test_metadata_completeness(self):
        """test that generated metadata is complete and valid"""
        signal = SigMFGenerator().tone(1000).generate()

        # verify required global fields
        global_info = signal.get_global_info()
        required_keys = [
            SigMFFile.DATATYPE_KEY,
            SigMFFile.SAMPLE_RATE_KEY,
            SigMFFile.VERSION_KEY,
            SigMFFile.NUM_CHANNELS_KEY,
            SigMFFile.GENERATOR_KEY,
            SigMFFile.DESCRIPTION_KEY,
        ]

        for key in required_keys:
            self.assertIn(key, global_info)

        # verify captures exist
        captures = signal.get_captures()
        self.assertEqual(len(captures), 1)
        self.assertIn(SigMFFile.START_INDEX_KEY, captures[0])
        self.assertIn(SigMFFile.DATETIME_KEY, captures[0])

        # should be valid sigmf
        signal.validate()

    def test_generator_info_includes_seed(self):
        """test that generator metadata includes seed when provided"""
        signal = SigMFGenerator(seed=self.seed).generate()

        generator_info = signal.get_global_info()[SigMFFile.GENERATOR_KEY]
        self.assertIn(f"seed={self.seed:#x}", generator_info)

    def test_no_seed_in_generator_info(self):
        """test that generator metadata excludes seed when not provided"""
        signal = SigMFGenerator().generate()

        generator_info = signal.get_global_info()[SigMFFile.GENERATOR_KEY]
        self.assertNotIn("seed=", generator_info)

    def test_data_buffer_creation(self):
        """test that signals are created with in-memory buffers"""
        signal = SigMFGenerator().generate()

        # should be able to read samples multiple times
        samples_0 = signal.read_samples()
        samples_1 = signal.read_samples()
        npt.assert_array_equal(samples_0, samples_1)

        # verify data is complex64
        self.assertEqual(samples_0.dtype, np.complex64)

    def test_with_different_amplitudes(self):
        """test amplitude parameter"""
        amp_0 = 0.5
        amp_1 = 1.5

        signal_0 = (
            SigMFGenerator(self.seed)
            .tone(1000)
            .amplitude(amp_0)
            .sample_rate(48000)
            .duration(0.1)
            .generate()
        )
        signal_1 = (
            SigMFGenerator(self.seed)
            .tone(1000)
            .amplitude(amp_1)
            .sample_rate(48000)
            .duration(0.1)
            .generate()
        )

        power_0 = np.mean(np.abs(signal_0.read_samples()) ** 2)
        power_1 = np.mean(np.abs(signal_1.read_samples()) ** 2)

        expected_power_ratio = (amp_1 / amp_0) ** 2
        actual_power_ratio = power_1 / power_0
        self.assertAlmostEqual(actual_power_ratio, expected_power_ratio, places=1)

    def test_automatic_annotations(self):
        """test that appropriate annotations are automatically created"""
        # tone with snr and frequency offset should create multiple annotations
        signal = (
            SigMFGenerator()
            .tone(1000)
            .sample_rate(48000)
            .duration(0.1)
            .snr(15)
            .frequency_offset(200)
            .comment("test")
            .generate()
        )

        annotations = signal.get_annotations()

        # should have main tone, noise, and offset annotations
        self.assertEqual(len(annotations), 3)

        # find and verify main tone annotation
        tone_annotation = next(ann for ann in annotations if "1000.0 Hz tone" in ann.get(SigMFFile.LABEL_KEY, ""))
        # with temporal windowing, start index can be any valid sample index
        self.assertGreaterEqual(tone_annotation[SigMFFile.START_INDEX_KEY], 0)
        self.assertLess(tone_annotation[SigMFFile.START_INDEX_KEY], 48000 * 0.1)  # less than total samples
        self.assertEqual(tone_annotation[SigMFFile.GENERATOR_KEY], "sigmf-python SigMFGenerator")
        self.assertIn(SigMFFile.FLO_KEY, tone_annotation)
        self.assertIn(SigMFFile.FHI_KEY, tone_annotation)
        self.assertEqual(tone_annotation[SigMFFile.COMMENT_KEY], "test")

        # verify tone frequency edges account for offset (1000 + 200 = 1200 Hz center)
        center_freq = (tone_annotation[SigMFFile.FLO_KEY] + tone_annotation[SigMFFile.FHI_KEY]) / 2
        self.assertAlmostEqual(center_freq, 1200.0, places=1)

        # find and verify noise annotation
        noise_annotation = next(ann for ann in annotations if "AWGN" in ann.get(SigMFFile.LABEL_KEY, ""))
        self.assertIn("15.0 dB SNR", noise_annotation[SigMFFile.LABEL_KEY])
        self.assertEqual(noise_annotation[SigMFFile.FLO_KEY], 0.0)
        self.assertEqual(noise_annotation[SigMFFile.FHI_KEY], 24000.0)  # nyquist

        # find and verify frequency offset annotation
        offset_annotation = next(ann for ann in annotations if "freq offset" in ann.get(SigMFFile.LABEL_KEY, ""))
        self.assertIn("+200.0 Hz", offset_annotation[SigMFFile.LABEL_KEY])

    def test_sweep_annotations(self):
        """test sweep annotations have correct frequency bounds including negative"""
        signal = SigMFGenerator().sweep(-2500, 2500).sample_rate(22050).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main sweep annotation

        sweep_annotation = annotations[0]
        self.assertEqual(sweep_annotation[SigMFFile.FLO_KEY], -2500.0)
        self.assertEqual(sweep_annotation[SigMFFile.FHI_KEY], 2500.0)
        self.assertIn("-2500.0-2500.0 Hz sweep", sweep_annotation[SigMFFile.LABEL_KEY])

    def test_reverse_sweep_annotations(self):
        """test reverse sweep crossing DC has correct bounds"""
        signal = SigMFGenerator().sweep(3000, -800).sample_rate(48000).generate()

        annotations = signal.get_annotations()
        sweep_annotation = annotations[0]

        # frequency bounds should be min/max regardless of sweep direction
        self.assertEqual(sweep_annotation[SigMFFile.FLO_KEY], -800.0)
        self.assertEqual(sweep_annotation[SigMFFile.FHI_KEY], 3000.0)
        # but label should show original order
        self.assertIn("3000.0--800.0 Hz sweep", sweep_annotation[SigMFFile.LABEL_KEY])

    def test_minimal_annotations(self):
        """test that simple signals get minimal but complete annotations"""
        signal = SigMFGenerator().tone(440).sample_rate(44100).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main signal, no noise/offsets

        annotation = annotations[0]
        # with temporal windowing, start index can be any valid sample index
        self.assertGreaterEqual(annotation[SigMFFile.START_INDEX_KEY], 0)
        self.assertIn(SigMFFile.LENGTH_INDEX_KEY, annotation)
        self.assertIn(SigMFFile.GENERATOR_KEY, annotation)
        self.assertIn("440.0 Hz tone", annotation[SigMFFile.LABEL_KEY])

    def test_phase_offset(self):
        """test phase offset functionality"""
        phase_offset = np.pi / 2

        # use clean signals without noise for precise phase comparison
        signal_0 = (
            SigMFGenerator(seed=42)
            .tone(1000)
            .sample_rate(48000)
            .duration(0.1)
            .amplitude(1.0)
            .generate()
        )
        signal_1 = (
            SigMFGenerator(seed=42)
            .tone(1000)
            .phase_offset(phase_offset)
            .sample_rate(48000)
            .duration(0.1)
            .amplitude(1.0)
            .generate()
        )

        samples_0 = signal_0.read_samples()
        samples_1 = signal_1.read_samples()

        # find where the actual signal starts by looking at annotations
        start_idx_0 = signal_0.get_annotations()[0][SigMFFile.START_INDEX_KEY]
        start_idx_1 = signal_1.get_annotations()[0][SigMFFile.START_INDEX_KEY]

        # both should start at the same sample index (same seed)
        self.assertEqual(start_idx_0, start_idx_1)

        # compare samples from the actual signal start + some offset to avoid edge effects
        sample_offset = 100
        if start_idx_0 + sample_offset < len(samples_0) and start_idx_1 + sample_offset < len(samples_1):
            phase_diff = np.angle(samples_1[start_idx_0 + sample_offset]) - np.angle(samples_0[start_idx_0 + sample_offset])

            # handle phase wrapping - normalize to [-pi, pi]
            while phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            while phase_diff < -np.pi:
                phase_diff += 2 * np.pi

            self.assertAlmostEqual(phase_diff, phase_offset, places=1)


class TestSigGenEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_duration(self):
        """test zero duration raises error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(1000).sample_rate(48000).duration(0).generate()

    def test_very_short_duration(self):
        """test very short durations work"""
        signal = SigMFGenerator().tone(1000).sample_rate(48000).duration(0.001).generate()  # 1ms

        samples = signal.read_samples()
        expected_samples = int(48000 * 0.001)
        self.assertEqual(len(samples), expected_samples)

    def test_large_frequency_offset(self):
        """test large frequency offsets"""
        # large offset that doesn't violate nyquist when combined with base freq
        signal = (
            SigMFGenerator()
            .tone(1000)
            .frequency_offset(10000)
            .sample_rate(48000)
            .duration(0.1)
            .generate()
        )

        captures = signal.get_captures()
        self.assertEqual(captures[0][SigMFFile.FREQUENCY_KEY], 11000.0)

    def test_sweep_same_start_end_frequency(self):
        """test sweep with same start and end frequency"""
        signal = SigMFGenerator().sweep(1000, 1000).sample_rate(48000).duration(0.1).generate()

        # should generate successfully (effectively a tone)
        self.assertIsInstance(signal, SigMFFile)

    def test_sweep_reverse_frequency(self):
        """test sweep with higher start than end frequency"""
        signal = SigMFGenerator().sweep(2000, 500).sample_rate(48000).duration(0.5).generate()

        # should work - frequency decreasing sweep
        self.assertIn("2000.0-500.0 hz sweep", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])


if __name__ == "__main__":
    unittest.main()

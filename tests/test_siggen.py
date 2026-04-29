import sigmf
# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for signal generation utilities."""

import unittest

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
            SigMFGenerator()
            .tone(self.test_freq)
            .sample_rate(self.test_sample_rate)
            .duration(self.test_duration)
            .generate()
        )

        # verify metadata
        self.assertEqual(signal.sample_rate, self.test_sample_rate)
        self.assertEqual(signal.datatype, "cf32_le")
        self.assertIn("tone", signal.description)

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

    def test_reproducible(self):
        """test that same seed produces identical results"""
        # generate signal 3 times with same seed
        # only difference will be datetime when capure is created
        signal0 = SigMFGenerator(self.seed).generate()
        signal1 = SigMFGenerator(self.seed).generate()
        signal2 = SigMFGenerator(self.seed).generate()

        # set capture datetime identical
        for sig in [signal0, signal1, signal2]:
            sig.add_capture(0, {sigmf.DATETIME_KEY: "2026-01-01T00:00:00Z"})

        # compare metadata (which includes checksums)
        self.assertEqual(signal0, signal1)
        self.assertEqual(signal0, signal2)

    def test_sweep_generation(self):
        """test linear frequency sweep generation"""
        start_freq_hz = -500.0
        end_freq_hz = 2000.0
        signal = (
            SigMFGenerator()
            .sweep(start_freq_hz, end_freq_hz)
            .sample_rate(self.test_sample_rate)
            .duration(self.test_duration)
            .generate()
        )

        samples = signal.read_samples()
        expected_samples = int(self.test_sample_rate * self.test_duration)
        self.assertEqual(len(samples), expected_samples)
        self.assertTrue(np.iscomplexobj(samples))

        # verify energy spans the sweep bandwidth
        fft_samples = np.fft.fft(samples)
        fft_freqs = np.fft.fftfreq(len(samples), 1 / self.test_sample_rate)
        power_spectrum = np.abs(fft_samples) ** 2

        # check that significant energy exists in the sweep band
        in_band = (fft_freqs >= start_freq_hz) & (fft_freqs <= end_freq_hz)
        out_of_band = ~in_band
        in_band_power = np.mean(power_spectrum[in_band])
        out_of_band_power = np.mean(power_spectrum[out_of_band])
        self.assertGreater(in_band_power, out_of_band_power)

    def test_nominal_chaining(self):
        """test builder pattern method chaining"""
        signal = (
            SigMFGenerator()
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
        self.assertEqual(signal.get_global_info()[sigmf.AUTHOR_KEY], "test@example.com")
        self.assertEqual(signal.description, "test signal")

        # should have multiple annotations: main signal + noise + freq offset + phase offset
        annotations = signal.get_annotations()
        self.assertGreaterEqual(len(annotations), 3)  # at least main + noise + offsets

        # find main signal annotation (has comment)
        main_annotation = next(ann for ann in annotations if sigmf.COMMENT_KEY in ann)
        self.assertEqual(main_annotation[sigmf.COMMENT_KEY], "test comment")

        # verify there's a noise annotation
        noise_annotations = [ann for ann in annotations if "AWGN" in ann.get(sigmf.LABEL_KEY, "")]
        self.assertEqual(len(noise_annotations), 1)

        # verify there's a frequency offset annotation
        freq_offset_annotations = [ann for ann in annotations if "freq offset" in ann.get(sigmf.LABEL_KEY, "")]
        self.assertEqual(len(freq_offset_annotations), 1)

    def test_snr_noise_addition(self):
        """test that snr parameter adds appropriate noise"""
        # generate high snr and low snr signals
        clean_signal = SigMFGenerator(self.seed).snr(40).generate()
        noisy_signal = SigMFGenerator(self.seed).snr(10).generate()

        # noisy signal should have higher variance due to added noise
        clean_power = np.mean(np.abs(clean_signal[:]) ** 2)
        noisy_power = np.mean(np.abs(noisy_signal[:]) ** 2)

        # noisy signal should have more power due to added noise
        self.assertGreater(noisy_power, clean_power)

    def test_frequency_offset(self):
        """test frequency offset functionality"""
        base_freq = 1000.0
        offset_freq = 500.0

        signal = SigMFGenerator().tone(base_freq).frequency_offset(offset_freq).generate()

        # verify frequency in capture metadata includes offset
        captures = signal.get_captures()
        self.assertEqual(captures[0][sigmf.FREQUENCY_KEY], base_freq + offset_freq)

    def test_metadata_completeness(self):
        """test that generated metadata is complete and valid"""
        signal = SigMFGenerator().tone().generate()

        # verify required global fields
        global_info = signal.get_global_info()
        required_keys = [
            sigmf.DATATYPE_KEY,
            sigmf.SAMPLE_RATE_KEY,
            sigmf.VERSION_KEY,
            sigmf.NUM_CHANNELS_KEY,
            sigmf.RECORDER_KEY,
            sigmf.DESCRIPTION_KEY,
        ]

        for key in required_keys:
            self.assertIn(key, global_info)

        # verify captures exist
        captures = signal.get_captures()
        self.assertEqual(len(captures), 1)
        self.assertIn(sigmf.SAMPLE_START_KEY, captures[0])
        self.assertIn(sigmf.DATETIME_KEY, captures[0])

        # should be valid sigmf
        signal.validate()

    def test_recorder_info(self):
        """test that recorder metadata includes seed when provided and excludes it when not"""
        with_seed = SigMFGenerator(seed=self.seed).generate()
        recorder_info = with_seed.get_global_info()[sigmf.RECORDER_KEY]
        self.assertIn(f"seed={self.seed:#x}", recorder_info)

        without_seed = SigMFGenerator().generate()
        recorder_info = without_seed.get_global_info()[sigmf.RECORDER_KEY]
        self.assertNotIn("seed=", recorder_info)

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
        amp_low = 0.5
        amp_high = 1.5

        signal_low = SigMFGenerator(self.seed).amplitude(amp_low).generate()
        signal_high = SigMFGenerator(self.seed).amplitude(amp_high).generate()

        power_low = np.mean(np.abs(signal_low.read_samples()) ** 2)
        power_high = np.mean(np.abs(signal_high.read_samples()) ** 2)

        expected_power_ratio = (amp_high / amp_low) ** 2
        actual_power_ratio = power_high / power_low
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
        tone_annotation = next(ann for ann in annotations if "tone at 1000 Hz" in ann.get(sigmf.LABEL_KEY, ""))
        # with temporal windowing, start index can be any valid sample index
        self.assertGreaterEqual(tone_annotation[sigmf.SAMPLE_START_KEY], 0)
        self.assertLess(tone_annotation[sigmf.SAMPLE_START_KEY], 48000 * 0.1)  # less than total samples
        self.assertEqual(tone_annotation[sigmf.GENERATOR_KEY], "SigMFGenerator")
        self.assertIn(sigmf.FREQ_LOWER_EDGE_KEY, tone_annotation)
        self.assertIn(sigmf.FREQ_UPPER_EDGE_KEY, tone_annotation)
        self.assertEqual(tone_annotation[sigmf.COMMENT_KEY], "test")

        # verify tone frequency edges account for offset (1000 + 200 = 1200 Hz center)
        center_freq = (
            tone_annotation[sigmf.FREQ_LOWER_EDGE_KEY] + tone_annotation[sigmf.FREQ_UPPER_EDGE_KEY]
        ) / 2
        self.assertAlmostEqual(center_freq, 1200.0, places=1)

        # find and verify noise annotation
        noise_annotation = next(ann for ann in annotations if "AWGN" in ann.get(sigmf.LABEL_KEY, ""))
        self.assertIn("15.0 dB SNR", noise_annotation[sigmf.LABEL_KEY])
        self.assertEqual(noise_annotation[sigmf.FREQ_LOWER_EDGE_KEY], 0.0)
        self.assertEqual(noise_annotation[sigmf.FREQ_UPPER_EDGE_KEY], 24000.0)  # nyquist

        # find and verify frequency offset annotation
        offset_annotation = next(ann for ann in annotations if "freq offset" in ann.get(sigmf.LABEL_KEY, ""))
        self.assertIn("+200.0 Hz", offset_annotation[sigmf.LABEL_KEY])

    def test_sweep_annotations(self):
        """test sweep annotations have correct frequency bounds including negative"""
        signal = SigMFGenerator().sweep(-2500, 2500).sample_rate(22050).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main sweep annotation

        sweep_annotation = annotations[0]
        self.assertEqual(sweep_annotation[sigmf.FREQ_LOWER_EDGE_KEY], -2500.0)
        self.assertEqual(sweep_annotation[sigmf.FREQ_UPPER_EDGE_KEY], 2500.0)
        self.assertIn("sweep from -2500 to 2500 Hz", sweep_annotation[sigmf.LABEL_KEY])

    def test_reverse_sweep_annotations(self):
        """test reverse sweep crossing DC has correct bounds"""
        signal = SigMFGenerator().sweep(3000, -800).sample_rate(48000).generate()

        annotations = signal.get_annotations()
        sweep_annotation = annotations[0]

        # frequency bounds should be min/max regardless of sweep direction
        self.assertEqual(sweep_annotation[sigmf.FREQ_LOWER_EDGE_KEY], -800.0)
        self.assertEqual(sweep_annotation[sigmf.FREQ_UPPER_EDGE_KEY], 3000.0)
        # but label should show original order
        self.assertIn("sweep from 3000 to -800 Hz", sweep_annotation[sigmf.LABEL_KEY])

    def test_minimal_annotations(self):
        """test that simple signals get minimal but complete annotations"""
        signal = SigMFGenerator().tone(440).sample_rate(44100).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main signal, no noise/offsets

        annotation = annotations[0]
        # with temporal windowing, start index can be any valid sample index
        self.assertGreaterEqual(annotation[sigmf.SAMPLE_START_KEY], 0)
        self.assertIn(sigmf.SAMPLE_COUNT_KEY, annotation)
        self.assertIn(sigmf.GENERATOR_KEY, annotation)
        self.assertIn("tone at 440 Hz", annotation[sigmf.LABEL_KEY])

    def test_phase_offset(self):
        """test phase offset functionality"""
        phase_offset = np.pi / 2

        # use clean signals without noise for precise phase comparison
        signal_0 = SigMFGenerator(seed=42).tone().generate()
        signal_1 = SigMFGenerator(seed=42).tone().phase_offset(phase_offset).generate()

        # tone annotations are last after sorting (full-signal annotations start at 0)
        start_idx_0 = signal_0.get_annotations()[0][sigmf.SAMPLE_START_KEY]
        start_idx_1 = signal_1.get_annotations()[0][sigmf.SAMPLE_START_KEY]

        # both should start at the same sample index (same seed)
        self.assertEqual(start_idx_0, start_idx_1)

        # compare samples from the actual signal start + some offset to avoid edge effects
        sample_offset = 100
        if start_idx_0 + sample_offset < len(signal_0) and start_idx_1 + sample_offset < len(signal_1):
            phase_diff = np.angle(signal_1[start_idx_0 + sample_offset]) - np.angle(
                signal_0[start_idx_0 + sample_offset]
            )

            # normalize to [-pi, pi]
            phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

            self.assertAlmostEqual(phase_diff, phase_offset, places=1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_duration(self):
        """test zero duration raises error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().duration(0).generate()

    def test_negative_duration(self):
        """test negative duration raises error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().duration(-1.0).generate()

    def test_negative_sample_rate(self):
        """test negative sample rate raises error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sample_rate(-8000).generate()

    def test_tone_nyquist_validation(self):
        """test tone frequency exceeding nyquist raises error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(5000).sample_rate(8000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().tone(-5000).sample_rate(8000).generate()

    def test_sweep_nyquist_validation(self):
        """test sweep frequencies exceeding nyquist raise error"""
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(1000, 5000).sample_rate(8000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(5000, 1000).sample_rate(8000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(1000, -5000).sample_rate(8000).generate()
        with self.assertRaises(SigMFGeneratorError):
            SigMFGenerator().sweep(-5000, 1000).sample_rate(8000).generate()

    def test_sweep_same_start_end_frequency(self):
        """test sweep with same start and end frequency"""
        # should generate successfully (effectively a tone)
        SigMFGenerator().sweep(333, 333).sample_rate(8000).duration(0.1).generate()


if __name__ == "__main__":
    unittest.main()

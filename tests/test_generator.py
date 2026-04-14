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
from sigmf.generate import SigMFGenerator


class TestSigMFGenerator(unittest.TestCase):
    """Test SigMFGenerator signal generation."""

    def setUp(self):
        """setup test fixtures"""
        self.test_seed = 0xDEADBEEF
        self.test_sample_rate = 48000
        self.test_duration = 1.0
        self.test_freq = 1000.0

    def test_deterministic_tone_generation(self):
        """test deterministic tone generation with specified parameters"""
        gen = SigMFGenerator(seed=self.test_seed)
        signal = gen.tone(self.test_freq).sample_rate(self.test_sample_rate).duration(self.test_duration).generate()

        # verify object type
        self.assertIsInstance(signal, SigMFFile)

        # verify metadata
        self.assertEqual(signal.sample_rate, self.test_sample_rate)
        self.assertEqual(signal.get_global_info()[SigMFFile.DATATYPE_KEY], "cf32_le")
        self.assertIn("1000.0 hz tone", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

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
        dominant_freq = abs(fft_freqs[dominant_freq_idx])
        self.assertAlmostEqual(dominant_freq, self.test_freq, delta=10)  # within 10 hz

    def test_random_tone_generation(self):
        """test random tone generation"""
        gen = SigMFGenerator(seed=self.test_seed)
        signal = gen.tone().generate()

        # should have reasonable defaults
        samples = signal.read_samples()
        self.assertGreater(len(samples), 1000)  # at least 0.1s at min sample rate
        self.assertIsInstance(signal, SigMFFile)
        self.assertIn("hz tone", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

    def test_reproducible_random_generation(self):
        """test that same seed produces identical results"""
        seed = 42

        # generate signal 3 times with same seed
        signal0 = SigMFGenerator(seed=seed).tone().generate()
        signal1 = SigMFGenerator(seed=seed).tone().generate()
        signal2 = SigMFGenerator(seed=seed).tone().generate()

        # compute checksums of the sample data
        import hashlib

        samples0 = signal0.read_samples().tobytes()
        samples1 = signal1.read_samples().tobytes()
        samples2 = signal2.read_samples().tobytes()

        hash0 = hashlib.sha256(samples0).hexdigest()
        hash1 = hashlib.sha256(samples1).hexdigest()
        hash2 = hashlib.sha256(samples2).hexdigest()

        # all hashes should be identical
        self.assertEqual(hash0, hash1)
        self.assertEqual(hash0, hash2)

    def test_sweep_generation(self):
        """test linear frequency sweep generation"""
        start_freq = 500.0
        end_freq = 2000.0

        gen = SigMFGenerator(seed=self.test_seed)
        signal = (
            gen.sweep(start_freq, end_freq).sample_rate(self.test_sample_rate).duration(self.test_duration).generate()
        )

        # verify metadata
        self.assertIn("500.0-2000.0 hz sweep", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])

        # verify signal properties
        samples = signal.read_samples()
        expected_samples = int(self.test_sample_rate * self.test_duration)
        self.assertEqual(len(samples), expected_samples)
        self.assertTrue(np.iscomplexobj(samples))

    def test_nominal_chaining(self):
        """test builder pattern method chaining"""
        signal = (
            SigMFGenerator(seed=self.test_seed)
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
        clean_gen = SigMFGenerator(seed=self.test_seed)
        clean_signal = clean_gen.tone(1000).sample_rate(self.test_sample_rate).duration(0.1).generate()

        noisy_gen = SigMFGenerator(seed=self.test_seed)
        noisy_signal = noisy_gen.tone(1000).sample_rate(self.test_sample_rate).duration(0.1).snr(10).generate()

        clean_samples = clean_signal.read_samples()
        noisy_samples = noisy_signal.read_samples()

        # noisy signal should have higher variance due to added noise
        clean_power = np.mean(np.abs(clean_samples) ** 2)
        noisy_power = np.mean(np.abs(noisy_samples) ** 2)

        # noisy signal should have more power due to added noise
        self.assertGreater(noisy_power, clean_power)

    def test_frequency_offset(self):
        """test frequency offset functionality"""
        base_freq = 1000.0
        offset_freq = 500.0

        gen = SigMFGenerator(seed=self.test_seed)
        signal = (
            gen.tone(base_freq)
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
        gen = SigMFGenerator()

        # should raise error if no signal type specified
        with self.assertRaises(SigMFGeneratorError):
            gen.generate()

        # should raise error for tone frequency exceeding nyquist
        with self.assertRaises(SigMFGeneratorError):
            gen.tone(30000).sample_rate(48000).duration(1.0).generate()

        # should raise error for negative duration
        with self.assertRaises(SigMFGeneratorError):
            gen.tone(1000).sample_rate(48000).duration(-1.0).generate()

        # should raise error for negative sample rate
        with self.assertRaises(SigMFGeneratorError):
            gen.tone(1000).sample_rate(-48000).duration(1.0).generate()

    def test_sweep_parameter_validation(self):
        """test sweep-specific parameter validation"""
        gen = SigMFGenerator()

        # sweep frequencies exceeding nyquist should raise error
        with self.assertRaises(SigMFGeneratorError):
            gen.sweep(1000, 30000).sample_rate(48000).duration(1.0).generate()

        with self.assertRaises(SigMFGeneratorError):
            gen.sweep(30000, 1000).sample_rate(48000).duration(1.0).generate()

    def test_random_parameters_reasonable(self):
        """test that random parameters are within reasonable ranges"""
        gen = SigMFGenerator(seed=42)
        signal = gen.tone().generate()

        # check sample rate is from common rates
        sample_rate = signal.sample_rate
        common_rates = [8000, 22050, 44100, 48000, 96000, 192000, 1e6, 2e6]
        self.assertIn(sample_rate, common_rates)

        # check duration is reasonable (0.1s to 5s)
        samples = signal.read_samples()
        actual_duration = len(samples) / sample_rate
        self.assertGreaterEqual(actual_duration, 0.1)
        self.assertLessEqual(actual_duration, 5.0)

    def test_random_sweep_parameters(self):
        """test random sweep parameter generation"""
        gen = SigMFGenerator(seed=42)
        signal = gen.sweep().generate()  # no parameters specified

        # should successfully generate
        self.assertIsInstance(signal, SigMFFile)

        # description should indicate it's a sweep
        desc = signal.get_global_info()[SigMFFile.DESCRIPTION_KEY]
        self.assertIn("sweep", desc)

        # should have start and end frequencies in description
        self.assertIn("-", desc)  # should have start-end format

    def test_metadata_completeness(self):
        """test that generated metadata is complete and valid"""
        gen = SigMFGenerator(seed=self.test_seed)
        signal = gen.tone(1000).sample_rate(48000).duration(1.0).generate()

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
        seed = 12345
        gen = SigMFGenerator(seed=seed)
        signal = gen.tone(1000).sample_rate(48000).duration(0.1).generate()

        generator_info = signal.get_global_info()[SigMFFile.GENERATOR_KEY]
        self.assertIn(f"seed={seed}", generator_info)

    def test_no_seed_in_generator_info(self):
        """test that generator metadata excludes seed when not provided"""
        gen = SigMFGenerator()  # no seed
        signal = gen.tone(1000).sample_rate(48000).duration(0.1).generate()

        generator_info = signal.get_global_info()[SigMFFile.GENERATOR_KEY]
        self.assertNotIn("seed=", generator_info)

    def test_data_buffer_creation(self):
        """test that signals are created with in-memory buffers"""
        gen = SigMFGenerator(seed=self.test_seed)
        signal = gen.tone(1000).sample_rate(48000).duration(0.1).generate()

        # should be able to read samples multiple times
        samples1 = signal.read_samples()
        samples2 = signal.read_samples()
        npt.assert_array_equal(samples1, samples2)

        # verify data is complex64
        self.assertEqual(samples1.dtype, np.complex64)

    def test_with_different_amplitudes(self):
        """test amplitude parameter"""
        amp1 = 0.5
        amp2 = 1.5

        gen1 = SigMFGenerator(seed=42)
        signal1 = gen1.tone(1000).amplitude(amp1).sample_rate(48000).duration(0.1).generate()

        gen2 = SigMFGenerator(seed=42)
        signal2 = gen2.tone(1000).amplitude(amp2).sample_rate(48000).duration(0.1).generate()

        samples1 = signal1.read_samples()
        samples2 = signal2.read_samples()

        # samples2 should have higher amplitude
        power1 = np.mean(np.abs(samples1) ** 2)
        power2 = np.mean(np.abs(samples2) ** 2)

        expected_power_ratio = (amp2 / amp1) ** 2
        actual_power_ratio = power2 / power1
        self.assertAlmostEqual(actual_power_ratio, expected_power_ratio, places=1)

    def test_automatic_annotations(self):
        """test that appropriate annotations are automatically created"""
        # tone with snr and frequency offset should create multiple annotations
        gen = SigMFGenerator(seed=42)
        signal = (
            gen.tone(1000).sample_rate(48000).duration(0.1).snr(15).frequency_offset(200).comment("test").generate()
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
        """test sweep annotations have correct frequency bounds"""
        gen = SigMFGenerator(seed=42)
        signal = gen.sweep(500, 2500).sample_rate(22050).duration(0.1).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main sweep annotation

        sweep_annotation = annotations[0]
        self.assertEqual(sweep_annotation[SigMFFile.FLO_KEY], 500.0)
        self.assertEqual(sweep_annotation[SigMFFile.FHI_KEY], 2500.0)
        self.assertIn("500.0-2500.0 Hz sweep", sweep_annotation[SigMFFile.LABEL_KEY])

    def test_reverse_sweep_annotations(self):
        """test reverse sweep (high to low freq) has correct bounds"""
        gen = SigMFGenerator(seed=42)
        signal = gen.sweep(3000, 800).sample_rate(48000).duration(0.1).generate()

        annotations = signal.get_annotations()
        sweep_annotation = annotations[0]

        # frequency bounds should be min/max regardless of sweep direction
        self.assertEqual(sweep_annotation[SigMFFile.FLO_KEY], 800.0)
        self.assertEqual(sweep_annotation[SigMFFile.FHI_KEY], 3000.0)
        # but label should show original order
        self.assertIn("3000.0-800.0 Hz sweep", sweep_annotation[SigMFFile.LABEL_KEY])

    def test_minimal_annotations(self):
        """test that simple signals get minimal but complete annotations"""
        gen = SigMFGenerator(seed=42)
        signal = gen.tone(440).sample_rate(44100).duration(0.1).generate()

        annotations = signal.get_annotations()
        self.assertEqual(len(annotations), 1)  # just main signal, no noise/offsets

        annotation = annotations[0]
        # with temporal windowing, start index can be any valid sample index
        self.assertGreaterEqual(annotation[SigMFFile.START_INDEX_KEY], 0)
        self.assertLess(annotation[SigMFFile.START_INDEX_KEY], 44100 * 0.1)  # less than total samples
        self.assertIn(SigMFFile.LENGTH_INDEX_KEY, annotation)
        self.assertIn(SigMFFile.GENERATOR_KEY, annotation)
        self.assertIn("440.0 Hz tone", annotation[SigMFFile.LABEL_KEY])

    def test_phase_offset(self):
        """test phase offset functionality"""
        phase_offset = np.pi / 2

        # use clean signals without noise for precise phase comparison
        gen1 = SigMFGenerator(seed=42)
        signal1 = gen1.tone(1000).sample_rate(48000).duration(0.1).amplitude(1.0).generate()

        gen2 = SigMFGenerator(seed=42)
        signal2 = gen2.tone(1000).phase_offset(phase_offset).sample_rate(48000).duration(0.1).amplitude(1.0).generate()

        samples1 = signal1.read_samples()
        samples2 = signal2.read_samples()

        # find where the actual signal starts by looking at annotations
        annotations1 = signal1.get_annotations()
        annotations2 = signal2.get_annotations()

        start_idx1 = annotations1[0][SigMFFile.START_INDEX_KEY]
        start_idx2 = annotations2[0][SigMFFile.START_INDEX_KEY]

        # both should start at the same sample index (same seed)
        self.assertEqual(start_idx1, start_idx2)

        # compare samples from the actual signal start + some offset to avoid edge effects
        sample_offset = 100
        if start_idx1 + sample_offset < len(samples1) and start_idx2 + sample_offset < len(samples2):
            phase_diff = np.angle(samples2[start_idx1 + sample_offset]) - np.angle(samples1[start_idx1 + sample_offset])

            # handle phase wrapping - normalize to [-pi, pi]
            while phase_diff > np.pi:
                phase_diff -= 2 * np.pi
            while phase_diff < -np.pi:
                phase_diff += 2 * np.pi

            self.assertAlmostEqual(phase_diff, phase_offset, places=1)


class TestSigMFGeneratorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_zero_duration(self):
        """test zero duration raises error"""
        gen = SigMFGenerator()
        with self.assertRaises(SigMFGeneratorError):
            gen.tone(1000).sample_rate(48000).duration(0).generate()

    def test_very_short_duration(self):
        """test very short durations work"""
        gen = SigMFGenerator()
        signal = gen.tone(1000).sample_rate(48000).duration(0.001).generate()  # 1ms

        samples = signal.read_samples()
        expected_samples = int(48000 * 0.001)
        self.assertEqual(len(samples), expected_samples)

    def test_large_frequency_offset(self):
        """test large frequency offsets"""
        gen = SigMFGenerator()
        # large offset that doesn't violate nyquist when combined with base freq
        signal = gen.tone(1000).frequency_offset(10000).sample_rate(48000).duration(0.1).generate()

        captures = signal.get_captures()
        self.assertEqual(captures[0][SigMFFile.FREQUENCY_KEY], 11000.0)

    def test_sweep_same_start_end_frequency(self):
        """test sweep with same start and end frequency"""
        gen = SigMFGenerator()
        signal = gen.sweep(1000, 1000).sample_rate(48000).duration(0.1).generate()

        # should generate successfully (effectively a tone)
        self.assertIsInstance(signal, SigMFFile)

    def test_sweep_reverse_frequency(self):
        """test sweep with higher start than end frequency"""
        gen = SigMFGenerator()
        signal = gen.sweep(2000, 500).sample_rate(48000).duration(0.5).generate()

        # should work - frequency decreasing sweep
        self.assertIn("2000.0-500.0 hz sweep", signal.get_global_info()[SigMFFile.DESCRIPTION_KEY])


if __name__ == "__main__":
    unittest.main()

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Simple signal generator utilities for SigMF."""

import io
from typing import Optional

import numpy as np

from .error import SigMFGeneratorError
from .sigmffile import SigMFFile
from .utils import get_data_type_str, get_sigmf_iso8601_datetime_now


class SigMFGenerator:
    """
    Builder pattern class for generating synthetic RF signals as SigMF files.

    Supports deterministic generation (with specified parameters) and random
    generation (parameterless methods with seed-controlled randomness).

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducible signal generation.

    Examples
    --------
    >>> # deterministic 1khz tone
    >>> gen = SigMFGenerator(seed=42)
    >>> signal = gen.tone(1000).sample_rate(48000).duration(1.0).generate()

    >>> # multiple tones combined
    >>> signal = gen.tone(1000).tone(1500).tone(2000).generate()

    >>> # tone plus sweep
    >>> signal = SigMFGenerator().sample_rate(100e3).tone(440).sweep(1000, 5000).duration(0.5).generate()
    """

    def __init__(self, seed: Optional[int] = None):
        # random state for reproducible generation across arch / platforms
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        # signal components (list of dicts)
        self._signal_components = []

        # signal configuration
        self._sample_rate_hz = None
        self._duration_s = None
        self._amplitude = 1.0
        self._snr_db = None
        self._frequency_offset_hz = 0.0
        self._phase_offset_rad = 0.0

        # metadata
        self._author = None
        self._description = None
        self._comment = None

    def tone(self, frequency_hz: Optional[float] = None, amplitude: Optional[float] = None):
        """
        Add a sinusoidal tone to the signal.

        Parameters
        ----------
        frequency_hz : float, optional
            Tone frequency in Hz. If None, will be randomly generated.
        amplitude : float, optional
            Tone amplitude (linear scale). If None, uses default amplitude.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        component = {"type": "tone"}
        if frequency_hz is not None:
            component["frequency_hz"] = float(frequency_hz)
        if amplitude is not None:
            component["amplitude"] = float(amplitude)
        self._signal_components.append(component)
        return self

    def sweep(
        self,
        start_frequency_hz: Optional[float] = None,
        end_frequency_hz: Optional[float] = None,
        amplitude: Optional[float] = None,
    ):
        """
        Add a linear frequency sweep to the signal.

        Parameters
        ----------
        start_frequency_hz : float, optional
            Starting frequency in Hz. If None, will be randomly generated.
        end_frequency_hz : float, optional
            Ending frequency in Hz. If None, will be randomly generated.
        amplitude : float, optional
            Sweep amplitude (linear scale). If None, uses default amplitude.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        component = {"type": "sweep"}
        if start_frequency_hz is not None:
            component["start_frequency_hz"] = float(start_frequency_hz)
        if end_frequency_hz is not None:
            component["end_frequency_hz"] = float(end_frequency_hz)
        if amplitude is not None:
            component["amplitude"] = float(amplitude)
        self._signal_components.append(component)
        return self

    def sample_rate(self, rate_hz: float):
        """
        Set sample rate.

        Parameters
        ----------
        rate_hz : float
            Sample rate in Hz.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._sample_rate_hz = float(rate_hz)
        return self

    def duration(self, duration_s: float):
        """
        Set signal duration.

        Parameters
        ----------
        duration_s : float
            Duration in seconds.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._duration_s = float(duration_s)
        return self

    def amplitude(self, amplitude: float):
        """
        Set signal amplitude.

        Parameters
        ----------
        amplitude : float
            Signal amplitude (linear scale).

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._amplitude = float(amplitude)
        return self

    def snr(self, snr_db: float):
        """
        Add white gaussian noise at specified snr.

        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._snr_db = float(snr_db)
        return self

    def frequency_offset(self, offset_hz: float):
        """
        Add frequency offset to signal.

        Parameters
        ----------
        offset_hz : float
            Frequency offset in Hz.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._frequency_offset_hz = float(offset_hz)
        return self

    def phase_offset(self, offset_rad: float):
        """
        Add phase offset to signal.

        Parameters
        ----------
        offset_rad : float
            Phase offset in radians.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._phase_offset_rad = float(offset_rad)
        return self

    def author(self, author: str):
        """
        Set author metadata.

        Parameters
        ----------
        author : str
            Author name/email.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._author = str(author)
        return self

    def description(self, description: str):
        """
        Set description metadata.

        Parameters
        ----------
        description : str
            Signal description.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._description = str(description)
        return self

    def comment(self, comment: str):
        """
        Set comment metadata.

        Parameters
        ----------
        comment : str
            Comment text.

        Returns
        -------
        SigMFGenerator
            Self for method chaining.
        """
        self._comment = str(comment)
        return self

    def generate(self) -> SigMFFile:
        """
        Generate the synthetic signal and return as sigmf file.

        Returns
        -------
        SigMFFile
            Generated signal file with metadata.

        Raises
        ------
        SigMFGeneratorError
            If required parameters are missing or invalid.
        """
        # validation and random parameter generation
        self._fill_random_parameters()
        self._validate_parameters()

        # generate signal samples
        samples = self._generate_samples()

        # create sigmf file with in-memory buffer
        data_buffer = io.BytesIO()
        data_buffer.write(samples.tobytes())
        data_buffer.seek(0)

        # build metadata
        metadata = self._build_metadata(samples)

        # create sigmf file object
        sigmf_file = SigMFFile(metadata=metadata)
        sigmf_file.set_data_file(data_buffer=data_buffer)

        return sigmf_file

    def _fill_random_parameters(self) -> None:
        """Fill unspecified parameters with random values."""
        # sample rates to choose from
        common_sample_rates = []
        # typical audio rates
        common_sample_rates += [8000, 22050, 44100, 48000, 96000, 192000]
        # typical SDR rates
        common_sample_rates += [250e3, 250e3, 400e3, 500e3, 800e3, 1.25e6, 2.5e6, 5e6, 6.25e6]

        # set random sample rate if not specified
        if self._sample_rate_hz is None:
            self._sample_rate_hz = float(self._rng.choice(common_sample_rates))

        # set random duration if not specified (0.1s to 5s)
        if self._duration_s is None:
            self._duration_s = self._rng.uniform(0.1, 5.0)

        # if no components specified, randomly generate some
        if len(self._signal_components) == 0:
            while True:
                if self._rng.random() < 0.5:
                    self._signal_components.append({"type": "tone"})
                else:
                    self._signal_components.append({"type": "sweep"})
                if self._rng.random() <= 0.2:
                    # E[N] = 1 / threshold -> 5 components on average
                    break

        # fill parameters for each signal component
        for component in self._signal_components:
            # add random timing for each component
            if "start_time_s" not in component:
                # random start time in first 80% of total duration
                max_start_time = self._duration_s * 0.8
                component["start_time_s"] = round(self._rng.uniform(0.0, max_start_time), 3)

            if "component_duration_s" not in component:
                # random duration from start time to end (minimum 10% of total duration)
                remaining_time = self._duration_s - component["start_time_s"]
                min_duration = min(self._duration_s * 0.1, remaining_time)
                component["component_duration_s"] = round(self._rng.uniform(min_duration, remaining_time), 3)

            # set amplitude if not specified
            if "amplitude" not in component:
                component["amplitude"] = self._amplitude

            if component["type"] == "tone":
                if "frequency_hz" not in component:
                    # random frequency across full baseband: -nyquist to +nyquist (excluding DC ±100 Hz)
                    nyquist = self._sample_rate_hz / 2
                    freq = self._rng.uniform(-nyquist + 100.0, nyquist - 100.0)
                    component["frequency_hz"] = round(freq, 1)

            elif component["type"] == "sweep":
                if "start_frequency_hz" not in component:
                    component["start_frequency_hz"] = round(self._rng.uniform(100.0, self._sample_rate_hz / 4 * 0.8), 1)
                if "end_frequency_hz" not in component:
                    start_freq = component["start_frequency_hz"]
                    # ensure end freq is different from start
                    if start_freq < self._sample_rate_hz / 4 * 0.5:
                        component["end_frequency_hz"] = round(
                            self._rng.uniform(start_freq * 1.5, self._sample_rate_hz / 4), 1
                        )
                    else:
                        component["end_frequency_hz"] = round(self._rng.uniform(100.0, start_freq * 0.7), 1)

    def _validate_parameters(self) -> None:
        """Validate current parameters."""
        if self._sample_rate_hz <= 0:
            raise SigMFGeneratorError(f"sample rate must be positive, got {self._sample_rate_hz}")

        if self._duration_s <= 0:
            raise SigMFGeneratorError(f"duration must be positive, got {self._duration_s}")

        # validate frequencies against nyquist limit
        nyquist = self._sample_rate_hz / 2
        for component in self._signal_components:
            frequencies_to_check = []

            if component["type"] == "tone":
                frequencies_to_check = [(component["frequency_hz"], "tone frequency")]

            elif component["type"] == "sweep":
                frequencies_to_check = [
                    (component["start_frequency_hz"], "start frequency"),
                    (component["end_frequency_hz"], "end frequency"),
                ]

            for freq, freq_name in frequencies_to_check:
                if abs(freq) >= nyquist:
                    raise SigMFGeneratorError(f"{freq_name} {freq} hz exceeds nyquist limit {nyquist} hz")

    def _generate_samples(self) -> np.ndarray:
        """Generate signal samples by combining all signal components with timing."""
        # calculate number of samples
        num_samples = int(self._sample_rate_hz * self._duration_s)
        time_samples = np.arange(num_samples, dtype=np.float64) / self._sample_rate_hz

        # initialize combined signal
        combined_signal = np.zeros(num_samples, dtype=np.complex128)

        # generate and sum each signal component with timing
        for component in self._signal_components:
            # calculate component timing in samples
            start_sample = int(component["start_time_s"] * self._sample_rate_hz)
            component_duration_samples = int(component["component_duration_s"] * self._sample_rate_hz)
            end_sample = min(start_sample + component_duration_samples, num_samples)

            if start_sample >= num_samples or end_sample <= start_sample:
                continue  # component is outside signal bounds

            # create time vector for this component only
            component_samples = end_sample - start_sample
            component_time = time_samples[start_sample:end_sample] - component["start_time_s"]

            if component["type"] == "tone":
                freq_hz = component["frequency_hz"]
                amplitude = component["amplitude"]
                component_signal = amplitude * np.exp(2j * np.pi * freq_hz * component_time)

            elif component["type"] == "sweep":
                start_freq = component["start_frequency_hz"]
                end_freq = component["end_frequency_hz"]
                amplitude = component["amplitude"]
                component_duration = component["component_duration_s"]

                # linear frequency sweep over component duration
                freq_slope = (end_freq - start_freq) / component_duration
                # integrate to get phase
                phase = 2 * np.pi * (start_freq * component_time + 0.5 * freq_slope * component_time**2)
                component_signal = amplitude * np.exp(1j * phase)

            # apply tapering to avoid clicks (5ms taper or 10% of component duration, whichever is smaller)
            taper_samples = min(int(0.005 * self._sample_rate_hz), component_samples // 10)
            if taper_samples > 1:
                # hann window tapering
                taper_window = np.hanning(2 * taper_samples)
                # apply fade-in
                component_signal[:taper_samples] *= taper_window[:taper_samples]
                # apply fade-out
                component_signal[-taper_samples:] *= taper_window[taper_samples:]

            # add this component to combined signal at correct time
            combined_signal[start_sample:end_sample] += component_signal

        # apply global frequency offset
        if self._frequency_offset_hz != 0:
            combined_signal *= np.exp(2j * np.pi * self._frequency_offset_hz * time_samples)

        # apply global phase offset
        if self._phase_offset_rad != 0:
            combined_signal *= np.exp(1j * self._phase_offset_rad)

        # add noise based on snr
        if self._snr_db is not None:
            signal_power = np.mean(np.abs(combined_signal) ** 2)
            noise_power = signal_power / (10 ** (self._snr_db / 10))

            # complex white gaussian noise
            noise_real = self._rng.normal(0, np.sqrt(noise_power / 2), num_samples)
            noise_imag = self._rng.normal(0, np.sqrt(noise_power / 2), num_samples)
            noise = noise_real + 1j * noise_imag

            combined_signal += noise

        # convert to complex64 for sigmf
        return combined_signal.astype(np.complex64)

    def _build_annotations(self, samples: np.ndarray) -> list:
        """Build annotations describing each signal component with timing."""
        annotations = []
        generator_name = "SigMFGenerator"

        # create annotation for each signal component
        for component in self._signal_components:
            # calculate component timing in samples
            start_sample = int(component["start_time_s"] * self._sample_rate_hz)
            component_duration_samples = int(component["component_duration_s"] * self._sample_rate_hz)
            end_sample = min(start_sample + component_duration_samples, len(samples))

            if start_sample >= len(samples) or end_sample <= start_sample:
                continue  # skip components outside signal bounds

            # base annotation common to all components
            base_annotation = {
                SigMFFile.START_INDEX_KEY: start_sample,
                SigMFFile.LENGTH_INDEX_KEY: end_sample - start_sample,
                SigMFFile.GENERATOR_KEY: generator_name,
            }

            if component["type"] == "tone":
                base_freq = component["frequency_hz"]
                total_freq = base_freq + self._frequency_offset_hz
                bandwidth = 2.0  # narrow bandwidth for tone

                base_annotation.update(
                    {
                        SigMFFile.FLO_KEY: total_freq - bandwidth / 2,
                        SigMFFile.FHI_KEY: total_freq + bandwidth / 2,
                        SigMFFile.LABEL_KEY: f"tone at {base_freq:.0f} Hz",
                    }
                )

            elif component["type"] == "sweep":
                start_freq = component["start_frequency_hz"] + self._frequency_offset_hz
                end_freq = component["end_frequency_hz"] + self._frequency_offset_hz

                base_annotation.update(
                    {
                        SigMFFile.FLO_KEY: min(start_freq, end_freq),
                        SigMFFile.FHI_KEY: max(start_freq, end_freq),
                        SigMFFile.LABEL_KEY: f"sweep from {component['start_frequency_hz']:.0f} to {component['end_frequency_hz']:.0f} Hz",
                    }
                )

            annotations.append(base_annotation)

        # add user comment to first component if provided
        if self._comment is not None and len(annotations) > 0:
            annotations[0][SigMFFile.COMMENT_KEY] = self._comment

        # helper function to create full-signal annotations
        def create_full_signal_annotation(label: str) -> dict:
            return {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.LENGTH_INDEX_KEY: len(samples),
                SigMFFile.GENERATOR_KEY: generator_name,
                SigMFFile.LABEL_KEY: label,
            }

        # noise annotation if snr was applied
        if self._snr_db is not None:
            noise_annotation = create_full_signal_annotation(f"AWGN {self._snr_db:.1f} dB SNR")
            noise_annotation.update(
                {
                    SigMFFile.FLO_KEY: 0.0,
                    SigMFFile.FHI_KEY: self._sample_rate_hz / 2,  # full nyquist bandwidth
                }
            )
            annotations.append(noise_annotation)

        # frequency offset annotation if applied
        if abs(self._frequency_offset_hz) > 0.1:  # only annotate non-trivial offsets
            offset_annotation = create_full_signal_annotation(f"freq offset {self._frequency_offset_hz:+.1f} Hz")
            annotations.append(offset_annotation)

        # phase offset annotation if applied
        if abs(self._phase_offset_rad) > 0.01:  # only annotate non-trivial offsets
            phase_deg = self._phase_offset_rad * 180 / np.pi
            phase_annotation = create_full_signal_annotation(f"phase offset {phase_deg:+.1f}°")
            annotations.append(phase_annotation)

        return annotations

    def _build_metadata(self, samples: np.ndarray) -> dict:
        """Build sigmf metadata dict."""
        # build description based on signal components
        if self._description is None:
            counts = {}
            for component in self._signal_components:
                counts[component["type"]] = counts.get(component["type"], 0) + 1

            parts = []
            for signal_type, count in counts.items():
                parts.append(f"{count} {signal_type}{'s' if count != 1 else ''}")

            if len(parts) == 0:
                desc = "synthetic signal"
            elif len(self._signal_components) == 1:
                desc = f"synthetic {parts[0][2:]}"  # strip "1 "
            else:
                desc = f"synthetic signal with {' and '.join(parts)}"

            if self._snr_db is not None:
                desc += f" at {self._snr_db:.1f} db snr"

            self._description = desc

        # build generator info
        generator_info = f"SigMFGenerator"
        if self._seed is not None:
            generator_info += f"(seed={self._seed:#x})"

        # create metadata structure
        global_info = {
            SigMFFile.DATATYPE_KEY: get_data_type_str(samples),
            SigMFFile.SAMPLE_RATE_KEY: self._sample_rate_hz,
            SigMFFile.NUM_CHANNELS_KEY: 1,
            SigMFFile.RECORDER_KEY: generator_info,
            SigMFFile.DESCRIPTION_KEY: self._description,
        }

        if self._author is not None:
            global_info[SigMFFile.AUTHOR_KEY] = self._author

        # create capture info
        capture_info = {
            SigMFFile.START_INDEX_KEY: 0,
            SigMFFile.DATETIME_KEY: get_sigmf_iso8601_datetime_now(),
        }

        # add frequency if there's a single dominant tone component
        tone_components = [c for c in self._signal_components if c["type"] == "tone"]
        if len(tone_components) == 1 and len(self._signal_components) == 1:
            dominant_freq = tone_components[0]["frequency_hz"] + self._frequency_offset_hz
            capture_info[SigMFFile.FREQUENCY_KEY] = dominant_freq

        # create annotations for signal components
        annotations = self._build_annotations(samples)

        return {
            SigMFFile.GLOBAL_KEY: global_info,
            SigMFFile.CAPTURE_KEY: [capture_info],
            SigMFFile.ANNOTATION_KEY: annotations,
        }

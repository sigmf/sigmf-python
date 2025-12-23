# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""converter for wav containers"""

import argparse
import logging
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .. import SigMFFile
from .. import __version__ as toolversion
from ..sigmffile import get_sigmf_filenames
from ..utils import SIGMF_DATETIME_ISO8601_FMT, get_data_type_str

log = logging.getLogger()

try:
    from scipy.io import wavfile
except ImportError:
    SCIPY_INSTALLED = False
else:
    SCIPY_INSTALLED = True


def wav_to_sigmf(
    wav_path: str,
    out_path: Optional[str] = None,
    create_archive: bool = False,
) -> SigMFFile:
    """
    Read a wav, optionally write a sigmf, return SigMFFile object.

    Raises
    ------
    wave.Error
        If the wav file is not PCM and Scipy is not installed.
    """
    wav_path = Path(wav_path)
    if SCIPY_INSTALLED:
        samp_rate, wav_data = wavfile.read(wav_path)
    else:
        with wave.open(str(wav_path), "rb") as wav_reader:
            n_channels = wav_reader.getnchannels()
            samp_width = wav_reader.getsampwidth()
            samp_rate = wav_reader.getframerate()
            n_frames = wav_reader.getnframes()
            raw_data = wav_reader.readframes(n_frames)
        np_dtype = f"int{samp_width * 8}"
        wav_data = np.frombuffer(raw_data, dtype=np_dtype).reshape(-1, n_channels)
    global_info = {
        SigMFFile.DATATYPE_KEY: get_data_type_str(wav_data),
        SigMFFile.DESCRIPTION_KEY: f"converted from {wav_path.name}",
        SigMFFile.NUM_CHANNELS_KEY: 1 if len(wav_data.shape) < 2 else wav_data.shape[1],
        SigMFFile.RECORDER_KEY: "Official SigMF WAV converter",
        SigMFFile.SAMPLE_RATE_KEY: samp_rate,
    }

    modify_time = wav_path.lstat().st_mtime
    wav_datetime = datetime.fromtimestamp(modify_time, tz=timezone.utc)

    capture_info = {
        SigMFFile.DATETIME_KEY: wav_datetime.strftime(SIGMF_DATETIME_ISO8601_FMT),
    }

    if out_path is None:
        base_path = wav_path.with_suffix(".sigmf")
    else:
        base_path = Path(out_path)

    filenames = get_sigmf_filenames(base_path)

    output_dir = filenames["meta_fn"].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if create_archive:
        # use temporary directory for data file when creating archive
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / filenames["data_fn"].name
            wav_data.tofile(data_path)

            meta = SigMFFile(data_file=data_path, global_info=global_info)
            meta.add_capture(0, metadata=capture_info)
            log.debug("created %r", meta)

            meta.tofile(filenames["archive_fn"], toarchive=True)
            log.info("wrote %s", filenames["archive_fn"])
    else:
        data_path = filenames["data_fn"]
        wav_data.tofile(data_path)

        meta = SigMFFile(data_file=data_path, global_info=global_info)
        meta.add_capture(0, metadata=capture_info)
        log.debug("created %r", meta)

        meta.tofile(filenames["meta_fn"], toarchive=False)
        log.info("wrote %s and %s", filenames["meta_fn"], filenames["data_fn"])

    return meta


def main() -> None:
    """
    Entry-point for sigmf_convert_wav
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=str, required=True, help="WAV path")
    parser.add_argument("-o", "--output", type=str, default=None, help="SigMF path")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "-a", "--archive", action="store_true", help="Save as SigMF archive instead of separate meta/data files."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s v{toolversion}")
    args = parser.parse_args()

    level_lut = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=level_lut[min(args.verbose, 2)])

    wav_path = Path(args.input)
    if args.output is None:
        args.output = wav_path.with_suffix(".sigmf")

    _ = wav_to_sigmf(wav_path=wav_path, out_path=args.output, create_archive=args.archive)


if __name__ == "__main__":
    main()

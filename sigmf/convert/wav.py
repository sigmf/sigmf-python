# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""converter for wav containers"""

import argparse
import getpass
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


def wav_to_sigmf(
    wav_path: str,
    out_path: Optional[str] = None,
    to_archive: bool = True,
    author: Optional[str] = None,
) -> SigMFFile:
    """
    Read a wav, write a sigmf, return SigMFFile object.

    Note: Can only read PCM wav files. Use scipy.io.wavefile for broader support.
    """
    wav_path = Path(wav_path)
    wav_stem = wav_path.stem
    with wave.open(str(wav_path), "rb") as wav_reader:
        n_channels = wav_reader.getnchannels()
        samp_width = wav_reader.getsampwidth()
        samp_rate = wav_reader.getframerate()
        n_frames = wav_reader.getnframes()
        raw_data = wav_reader.readframes(n_frames)
    np_dtype = f"int{samp_width * 8}"
    wav_data = np.frombuffer(raw_data, dtype=np_dtype).reshape(-1, n_channels)
    global_info = {
        SigMFFile.AUTHOR_KEY: getpass.getuser() if author is None else author,
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

    temp_dir = Path(tempfile.mkdtemp())
    if out_path is None:
        # extension will be changed
        out_path = Path(wav_stem)
    else:
        out_path = Path(out_path)
    filenames = get_sigmf_filenames(out_path)

    data_path = temp_dir / filenames["data_fn"]
    wav_data.tofile(data_path)

    meta = SigMFFile(data_file=data_path, global_info=global_info)
    meta.add_capture(0, metadata=capture_info)
    log.debug("created %r", meta)

    arc_path = filenames["archive_fn"]
    meta.tofile(arc_path, toarchive=True)
    log.info("wrote %s", arc_path)
    return meta


def main() -> None:
    """
    Entry-point for sigmf_convert_wav
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="wav path")
    parser.add_argument("--author", type=str, default=None, help=f"set {SigMFFile.AUTHOR_KEY} metadata")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"%(prog)s v{toolversion}")
    args = parser.parse_args()

    level_lut = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=level_lut[min(args.verbose, 2)])

    _ = wav_to_sigmf(
        wav_path=args.input,
        author=args.author,
    )


if __name__ == "__main__":
    main()

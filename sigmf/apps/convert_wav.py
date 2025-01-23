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
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Optional

from scipy.io import wavfile

from .. import SigMFFile
from .. import __version__ as toolversion
from ..sigmffile import get_sigmf_filenames
from ..utils import SIGMF_DATETIME_ISO8601_FMT, get_data_type_str

log = logging.getLogger()


def convert_wav(
    wav_path: str,
    out_path: Optional[str] = None,
    author: Optional[str] = None,
) -> PathLike:
    """
    Read a wav and write a sigmf archive.
    """
    wav_path = Path(wav_path)
    wav_stem = wav_path.stem
    samp_rate, wav_data = wavfile.read(wav_path)

    global_info = {
        SigMFFile.AUTHOR_KEY: getpass.getuser() if author is None else author,
        SigMFFile.DATATYPE_KEY: get_data_type_str(wav_data),
        SigMFFile.DESCRIPTION_KEY: f"converted from {wav_path.name}",
        SigMFFile.NUM_CHANNELS_KEY: 1 if len(wav_data.shape) < 2 else wav_data.shape[1],
        SigMFFile.RECORDER_KEY: "Official SigMF wav converter",
        SigMFFile.SAMPLE_RATE_KEY: samp_rate,
    }

    modify_time = wav_path.lstat().st_mtime
    wav_datetime = datetime.fromtimestamp(modify_time, tz=timezone.utc)

    capture_info = {
        SigMFFile.START_INDEX_KEY: 0,
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
    return arc_path


def main() -> None:
    """
    entry-point for sigmf_convert_wav
    """
    parser = argparse.ArgumentParser(description="Convert wav to sigmf archive.")
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

    _ = convert_wav(
        wav_path=args.input,
        author=args.author,
    )


if __name__ == "__main__":
    main()

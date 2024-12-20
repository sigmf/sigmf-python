# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""converter for wav containers"""

import argparse
import datetime
import getpass
import logging
import os
import pathlib
import tempfile

from scipy.io import wavfile

from .. import SigMFFile, __specification__
from .. import __version__ as toolversion
from .. import archive
from ..utils import get_data_type_str

log = logging.getLogger()


def convert_wav(input_wav_filename, archive_filename=None, start_datetime=None, author=None):
    """
    read a .wav and write a .sigmf archive
    """
    input_path = pathlib.Path(input_wav_filename)
    input_stem = input_path.stem
    samp_rate, wav_data = wavfile.read(input_wav_filename)

    global_info = {
        SigMFFile.AUTHOR_KEY: getpass.getuser() if author is None else author,
        SigMFFile.DATATYPE_KEY: get_data_type_str(wav_data),
        SigMFFile.DESCRIPTION_KEY: f"Converted from {input_wav_filename}",
        SigMFFile.NUM_CHANNELS_KEY: 1 if len(wav_data.shape) < 2 else wav_data.shape[1],
        SigMFFile.RECORDER_KEY: os.path.basename(__file__),
        SigMFFile.SAMPLE_RATE_KEY: samp_rate,
        SigMFFile.VERSION_KEY: __specification__,
    }

    if start_datetime is None:
        mtime = datetime.datetime.fromtimestamp(input_path.stat().st_mtime)
        start_datetime = mtime.isoformat() + "Z"

    capture_info = {SigMFFile.START_INDEX_KEY: 0}
    if start_datetime is not None:
        capture_info[SigMFFile.DATETIME_KEY] = start_datetime

    tmpdir = tempfile.mkdtemp()
    sigmf_data_filename = input_stem + archive.SIGMF_DATASET_EXT
    sigmf_data_path = os.path.join(tmpdir, sigmf_data_filename)
    wav_data.tofile(sigmf_data_path)

    meta = SigMFFile(data_file=sigmf_data_path, global_info=global_info)
    meta.add_capture(0, metadata=capture_info)

    if archive_filename is None:
        archive_filename = input_stem + archive.SIGMF_ARCHIVE_EXT
    meta.tofile(archive_filename, toarchive=True)
    return os.path.abspath(archive_filename)


def main():
    """
    entry-point for sigmf_convert_wav
    """
    parser = argparse.ArgumentParser(description="Convert .wav to .sigmf container.")
    parser.add_argument("input", type=str, help="Wavfile path")
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

    out_fname = convert_wav(
        input_wav_filename=args.input,
        author=args.author,
    )
    log.info(f"Write {out_fname}")


if __name__ == "__main__":
    main()

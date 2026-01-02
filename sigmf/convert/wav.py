# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""converter for wav containers"""

import argparse
import io
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


def _calculate_wav_ncd_bytes(wav_path: Path) -> tuple:
    """
    Calculate header_bytes and trailing_bytes for WAV NCD.

    Returns
    -------
    tuple
        (header_bytes, trailing_bytes)
    """
    # use wave module to get basic info
    with wave.open(str(wav_path), "rb") as wav_reader:
        n_channels = wav_reader.getnchannels()
        samp_width = wav_reader.getsampwidth()
        n_frames = wav_reader.getnframes()

    # calculate sample data size in bytes
    sample_bytes = n_frames * n_channels * samp_width
    file_size = wav_path.stat().st_size

    # parse WAV file structure to find data chunk
    with open(wav_path, "rb") as handle:
        # skip RIFF header (12 bytes: 'RIFF' + size + 'WAVE')
        handle.seek(12)
        header_bytes = 12

        # search for 'data' chunk
        while header_bytes < file_size:
            chunk_id = handle.read(4)
            if len(chunk_id) != 4:
                break
            chunk_size = int.from_bytes(handle.read(4), "little")

            if chunk_id == b"data":
                # found data chunk, header ends here
                header_bytes += 8  # include chunk_id and chunk_size
                break

            # skip this chunk
            header_bytes += 8 + chunk_size
            # ensure even byte boundary (WAV chunks are word-aligned)
            if chunk_size % 2:
                header_bytes += 1
            handle.seek(header_bytes)

    trailing_bytes = max(0, file_size - header_bytes - sample_bytes)
    return header_bytes, trailing_bytes


def wav_to_sigmf(
    wav_path: str,
    out_path: Optional[str] = None,
    create_archive: bool = False,
    create_ncd: bool = False,
) -> SigMFFile:
    """
    Read a wav, optionally write a sigmf, return SigMFFile object.

    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    out_path : str, optional
        Path to the output SigMF metadata file.
    create_archive : bool, optional
        When True, package output as a .sigmf archive.
    create_ncd : bool, optional
        When True, create Non-Conforming Dataset with header_bytes and trailing_bytes.

    Returns
    -------
    SigMFFile
        SigMF object, potentially as Non-Conforming Dataset.

    Raises
    ------
    wave.Error
        If the wav file cannot be read.
    """
    wav_path = Path(wav_path)

    # auto-enable NCD when no output path is specified
    if out_path is None:
        create_ncd = True

    # use built-in wave module exclusively for precise sample boundary detection
    with wave.open(str(wav_path), "rb") as wav_reader:
        n_channels = wav_reader.getnchannels()
        samp_width = wav_reader.getsampwidth()
        samp_rate = wav_reader.getframerate()
        n_frames = wav_reader.getnframes()

        # for NCD support, calculate precise byte boundaries
        if create_ncd:
            header_bytes, trailing_bytes = _calculate_wav_ncd_bytes(wav_path)
            log.debug(f"WAV NCD: header_bytes={header_bytes}, trailing_bytes={trailing_bytes}")

        # only read audio data if we're not creating NCD metadata-only
        wav_data = None  # initialize variable
        if create_ncd and out_path is None:
            # metadata-only NCD: don't read audio data
            pass
        else:
            # normal conversion: read the audio data
            raw_data = wav_reader.readframes(n_frames)

    np_dtype = f"int{samp_width * 8}"

    if wav_data is None:
        # for NCD metadata-only, create dummy sample to get datatype
        dummy_sample = np.array([0], dtype=np_dtype)
        datatype_str = get_data_type_str(dummy_sample)
    else:
        # normal case: process actual audio data
        wav_data = (
            np.frombuffer(raw_data, dtype=np_dtype).reshape(-1, n_channels)
            if n_channels > 1
            else np.frombuffer(raw_data, dtype=np_dtype)
        )
        datatype_str = get_data_type_str(wav_data)

    global_info = {
        SigMFFile.DATATYPE_KEY: datatype_str,
        SigMFFile.DESCRIPTION_KEY: f"converted from {wav_path.name}",
        SigMFFile.NUM_CHANNELS_KEY: n_channels,
        SigMFFile.RECORDER_KEY: "Official SigMF WAV converter",
        SigMFFile.SAMPLE_RATE_KEY: samp_rate,
    }

    modify_time = wav_path.lstat().st_mtime
    wav_datetime = datetime.fromtimestamp(modify_time, tz=timezone.utc)

    capture_info = {
        SigMFFile.DATETIME_KEY: wav_datetime.strftime(SIGMF_DATETIME_ISO8601_FMT),
    }

    if create_ncd:
        # NCD requires extra fields
        global_info[SigMFFile.TRAILING_BYTES_KEY] = trailing_bytes
        global_info[SigMFFile.DATASET_KEY] = wav_path.name
        capture_info[SigMFFile.HEADER_BYTES_KEY] = header_bytes

    # handle NCD case where no output files are created
    if create_ncd and out_path is None:
        # create metadata-only SigMF for NCD pointing to original file
        meta = SigMFFile(global_info=global_info, skip_checksum=True)
        meta.set_data_file(data_file=wav_path, offset=header_bytes, skip_checksum=True)
        meta.data_buffer = io.BytesIO()
        meta.add_capture(0, metadata=capture_info)
        log.debug("created NCD SigMF: %r", meta)
        return meta

    # if we get here, we need the actual audio data to create a new data file
    if wav_data is None:
        # need to read the audio data now for normal file creation
        with wave.open(str(wav_path), "rb") as wav_reader:
            raw_data = wav_reader.readframes(n_frames)
        wav_data = (
            np.frombuffer(raw_data, dtype=np_dtype).reshape(-1, n_channels)
            if n_channels > 1
            else np.frombuffer(raw_data, dtype=np_dtype)
        )

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
    parser.add_argument(
        "--ncd", action="store_true", help="Process as Non-Conforming Dataset and write .sigmf-meta only."
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

    _ = wav_to_sigmf(wav_path=wav_path, out_path=args.output, create_archive=args.archive, create_ncd=args.ncd)


if __name__ == "__main__":
    main()

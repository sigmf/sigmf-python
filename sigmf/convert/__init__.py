# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Convert non-SigMF recordings to SigMF format"""

from pathlib import Path

from ..error import SigMFConversionError


def get_magic_bytes(file_path: Path, count: int = 4, offset: int = 0) -> bytes:
    """
    Get magic bytes from a file to help identify file type.

    Parameters
    ----------
    file_path : Path
        Path to the file to read magic bytes from.
    count : int, optional
        Number of bytes to read. Default is 4.
    offset : int, optional
        Byte offset to start reading from. Default is 0.

    Returns
    -------
    bytes
        Magic bytes from the file.

    Raises
    ------
    SigMFConversionError
        If file cannot be read or is too small.
    """
    try:
        with open(file_path, "rb") as handle:
            handle.seek(offset)
            magic_bytes = handle.read(count)
            if len(magic_bytes) < count:
                raise SigMFConversionError(f"File {file_path} too small to read {count} magic bytes at offset {offset}")
            return magic_bytes
    except OSError as err:
        raise SigMFConversionError(f"Failed to read magic bytes from {file_path}: {err}") from err


def detect_converter(file_path: Path):
    """
    Detect the appropriate converter for a non-SigMF file.

    Parameters
    ----------
    file_path : Path
        Path to the file to detect.

    Returns
    -------
    str
        The converter name: "wav", "blue", or "signalhound"

    Raises
    ------
    SigMFConversionError
        If the file format is not supported or cannot be detected.
    """
    magic_bytes = get_magic_bytes(file_path, count=4, offset=0)

    if magic_bytes == b"RIFF":
        return "wav"

    elif magic_bytes == b"BLUE":
        return "blue"

    elif magic_bytes == b"<?xm":  # <?xml version="1.0" encoding="UTF-8"?>
        # Check if it's a Signal Hound Spike file
        # Skip XML declaration (40 bytes) and check for SignalHoundIQFile root element
        expanded_magic_bytes = get_magic_bytes(file_path, count=17, offset=40)
        if expanded_magic_bytes == b"SignalHoundIQFile":
            return "signalhound"
        else:
            raise SigMFConversionError(
                f"Unsupported XML file format. Root element: {expanded_magic_bytes}. "
                f"Expected SignalHoundIQFile for Signal Hound Spike files."
            )

    else:
        raise SigMFConversionError(
            f"Unsupported file format. Magic bytes: {magic_bytes}. "
            f"Supported formats for conversion are WAV, BLUE/Platinum, and Signal Hound Spike."
        )

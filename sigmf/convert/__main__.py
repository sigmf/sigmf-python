# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Unified converter for non-SigMF file formats"""

import argparse
import logging
import textwrap
from pathlib import Path

from .. import __version__ as toolversion
from ..error import SigMFConversionError
from ..utils import get_magic_bytes
from .blue import blue_to_sigmf
from .wav import wav_to_sigmf
from .signalhound import signalhound_to_sigmf



def main() -> None:
    """
    Unified entry-point for SigMF conversion of non-SigMF recordings.

    This command-line interface converts various non-SigMF file formats into SigMF-compliant datasets.
    It currently supports WAV and BLUE/Platinum file formats.
    The converter detects the file type based on magic bytes and invokes the appropriate conversion function.

    By default it will output a SigMF pair (.sigmf-meta and .sigmf-data).

    Converter Processing Pattern
    ----------------------------
    if out_path is None:
        create_ncd = True
    <create global_info and capture_info>
    if create_ncd:
        <create Non-Conforming Dataset (NCD) with .sigmf-meta only>
        if out_path:
            <write out_path.sigmf-meta>
        return SigMFFile
    if create_archive:
        with TemporaryDirectory() as temp_dir:
            <write .sigmf-data>
        <write out_path.sigmf>
    else:
        <write out_path.sigmf-data>
        <write out_path.sigmf-meta>
    return SigMFFile
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(main.__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog="sigmf_convert",
    )
    parser.add_argument("input", type=str, help="Input recording path")
    parser.add_argument("output", type=str, help="Output SigMF path (no extension)")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity level")
    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("-a", "--archive", action="store_true", help="Output .sigmf archive only")
    exclusive_group.add_argument(
        "--ncd", action="store_true", help="Output .sigmf-meta only and process as a Non-Conforming Dataset (NCD)"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s v{toolversion}")

    args = parser.parse_args()

    level_lut = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=level_lut[min(args.verbose, 2)])

    input_path = Path(args.input)
    output_path = Path(args.output)

    # for ncd check that input & output files are in same directory
    if args.ncd and input_path.parent.resolve() != output_path.parent.resolve():
        raise SigMFConversionError(
            f"NCD files must be in the same directory as input file. "
            f"Input: {input_path.parent.resolve()}, Output: {output_path.parent.resolve()}"
        )

    # check that the output path is a file and not a directory
    if output_path.is_dir():
        raise SigMFConversionError(f"Output path must be a filename, not a directory: {output_path}")

    # detect file type using magic bytes (same logic as fromfile())
    magic_bytes = get_magic_bytes(input_path, count=4, offset=0)

    if magic_bytes == b"RIFF":
        # WAV file
        _ = wav_to_sigmf(wav_path=input_path, out_path=output_path, create_archive=args.archive, create_ncd=args.ncd)

    elif magic_bytes == b"BLUE":
        # BLUE file
        _ = blue_to_sigmf(blue_path=input_path, out_path=output_path, create_archive=args.archive, create_ncd=args.ncd)

    ## TODO: Determine proper way to integrate Signal Hound files. 
    elif magic_bytes == b"<?xm": # <?xml version="1.0" encoding="UTF-8"?> <SignalHoundIQFile Version="1.0">
       # Signal Hound Spike 1.0 file
        # Of the 66 Byte string move 43 bytes in to skip the XML declaration
        # and get to the root element and take 18 chars for a more specific detection of Signal Hound Spike files
        expanded_magic_bytes = get_magic_bytes(input_path, count=17, offset=40)
        if expanded_magic_bytes == b"SignalHoundIQFile":
            _ = signalhound_to_sigmf(signalhound_path=input_path, out_path=output_path, create_archive=args.archive, create_ncd=args.ncd)
        else:
          raise SigMFConversionError(
            f"Unsupported XML file format. Expanded Magic bytes: {expanded_magic_bytes}. "
            f"Supported formats for conversion are WAV, BLUE/Platinum and Signal Hound Spike."
        )
    else:
        raise SigMFConversionError(
            f"Unsupported file format. Magic bytes: {magic_bytes}. "
            f"Supported formats for conversion are WAV, BLUE/Platinum and Signal Hound Spike."
        )


if __name__ == "__main__":
    main()

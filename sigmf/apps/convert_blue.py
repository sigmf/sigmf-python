# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
X-Midas BLUE File converter.
This script reads and parses the HCB (Header Control Block) and Extended Headers.
It supports different file types and extracts metadata accordingly.
Converts the extracted metadata into SigMF format.
"""

import argparse
import getpass
import logging
import os
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .. import SigMFFile
from .. import __version__ as toolversion
from ..error import SigMFConversionError
from ..sigmffile import get_sigmf_filenames
from ..utils import SIGMF_DATETIME_ISO8601_FMT

log = logging.getLogger()

# fmt: off
FIXED_LAYOUT = [
    # Fixed Header definitions: (key, offset, size, fmt, description) up to adjunct
    ("version",   0,   4,  "4s",   "Header version"),
    ("head_rep",  4,   4,  "4s",   "Header representation"),
    ("data_rep",  8,   4,  "4s",   "Data representation"),
    ("detached",  12,  4,  "i",    "Detached header"),
    ("protected", 16,  4,  "i",    "Protected from overwrite"),
    ("pipe",      20,  4,  "i",    "Pipe mode (N/A)"),
    ("ext_start", 24,  4,  "i",    "Extended header start (512-byte blocks)"),
    ("ext_size",  28,  4,  "i",    "Extended header size in bytes"),
    ("data_start",32,  8,  "d",    "Data start in bytes"),
    ("data_size", 40,  8,  "d",    "Data size in bytes"),
    ("type",      48,  4,  "i",    "File type code"),
    ("format",    52,  2,  "2s",   "2 Letter data format code"),
    ("flagmask",  54,  2,  "h",    "16-bit flagmask"),
    ("timecode",  56,  8,  "d",    "Time code field"),
    ("inlet",     64,  2,  "h",    "Inlet owner"),
    ("outlets",   66,  2,  "h",    "Number of outlets"),
    ("outmask",   68,  4,  "i",    "Outlet async mask"),
    ("pipeloc",   72,  4,  "i",    "Pipe location"),
    ("pipesize",  76,  4,  "i",    "Pipe size in bytes"),
    ("in_byte",   80,  8,  "d",    "Next input byte"),
    ("out_byte",  88,  8,  "d",    "Next out byte (cumulative)"),
    ("outbytes",  96,  64, "8d",   "Next out byte (each outlet)"),
    ("keylength", 160, 4,  "i",    "Length of keyword string"),
    ("keywords",  164, 92, "92s",  "User defined keyword string"),
    # Adjunct starts at byte 256 after this
]
# fmt: on

TYPE_MAP = {
    # Extended Header type map
    "B": (np.int8, 1),
    "I": (np.int16, 2),
    "L": (np.int32, 4),
    "X": (np.int64, 8),
    "F": (np.float32, 4),
    "D": (np.float64, 8),
    "A": (np.dtype("S1"), 1),
}

HEADER_SIZE_BYTES = 512
BLOCK_SIZE_BYTES = 512

NORMALIZATION_FACTORS = {
    # format : normalization factor
    "SB": 2**7 - 1,  # scalar 8-bit integer
    "SI": 2**15 - 1,  # scalar 16-bit integer
    "SL": 2**31 - 1,  # scalar 32-bit integer
    "CB": 2**7 - 1,  # complex 8-bit integer
    "CI": 2**15 - 1,  # complex 16-bit integer
    "CL": 2**31 - 1,  # complex 32-bit integer
}

# Data type configurations
DATA_TYPE_CONFIGS = {
    "CB": {"dtype": np.int8, "complex": True, "normalize": True},
    "CI": {"dtype": np.int16, "complex": True, "normalize": True},
    "CL": {"dtype": np.int32, "complex": True, "normalize": True},
    "CF": {"dtype": np.complex64, "complex": True, "normalize": False},
    "SB": {"dtype": np.int8, "complex": False, "normalize": True},
    "SI": {"dtype": np.int16, "complex": False, "normalize": True},
    "SL": {"dtype": np.int32, "complex": False, "normalize": True},
    "SX": {"dtype": np.int64, "complex": False, "normalize": False},
    "SF": {"dtype": np.float32, "complex": False, "normalize": False},
    "SD": {"dtype": np.float64, "complex": False, "normalize": False},
}

DATATYPE_MAP_BASE = {
    # S = Scalar
    "SB": "ri8",
    "SI": "ri16",
    "SL": "ri32",
    "SX": "ri64",
    "SF": "rf32",
    "SD": "rf64",
    # C = Complex
    "CB": "ci8",
    "CI": "ci16",
    "CL": "ci32",
    "CX": "ci64",
    "CF": "cf32",
    "CD": "cf32",  # FIXME: should be cf64? D should be double.
    # V = Vector (not supported)
    # Q = Quad (not supported)
}


def detect_endian(data, probe_fields=("data_size", "version")):
    """
    Detect endianness of a Bluefile header.

    TODO: Look at this code and see if can be improved and possibly simplified.

    Parameters
    ----------
    data : bytes
        Raw header data.
    probe_fields : tuple of str, optional
        Field names to test for sanity checks.

    Returns
    -------
    str
        "<" for little-endian or ">" for big-endian.

    Raises
    ------
    SigMFConversionError
        If the endianness is unexpected.
    """
    # TODO: handle both types of endianess 'EEEI' or IEEE and data rep and signal rep
    endianness = data[8:12].decode("ascii")
    if endianness not in ("EEEI", "IEEE"):
        raise SigMFConversionError(f"Unexpected endianness: {endianness}")

    for endian in ("<", ">"):
        ok = True
        for key, offset, size, fmt, _ in FIXED_LAYOUT:
            if key not in probe_fields:
                continue
            raw = data[offset : offset + size]
            try:
                val = struct.unpack(endian + fmt, raw)[0]
                # sanity checks
                MAX_DATA_SIZE_FACTOR = 100

                if key == "data_size":
                    if val <= 0 or val > len(data) * MAX_DATA_SIZE_FACTOR:
                        ok = False
                        break
                elif key == "version":
                    if not 0 < val < 10:  # expect small version number
                        ok = False
                        break
            except Exception:
                ok = False
                break
        if ok:
            return endian
    # fallback
    return "<"


def read_hcb(file_path):
    """
    Read Header Control Block (HCB) from BLUE file.

    First 256 bytes contains fixed header, followed by 256 bytes of adjunct header.

    Parameters
    ----------
    file_path : str
        Path to the Blue file.

    Returns
    -------
    h_fixed : dict
        Fixed Header
    h_keywords : dict
        Custom User Keywords
    h_adjunct : dict
        Adjunct Header

    Raises
    ------
    SigMFConversionError
        If header cannot be parsed.
    """
    with open(file_path, "rb") as handle:
        header_bytes = handle.read(256)

        endian = detect_endian(header_bytes)

        # fixed header fields
        h_fixed = {}
        for key, offset, size, fmt, _ in FIXED_LAYOUT:
            raw = header_bytes[offset : offset + size]
            try:
                val = struct.unpack(endian + fmt, raw)[0]
            except struct.error:
                raise SigMFConversionError(f"Failed to unpack field {key} with endian {endian}")
            if isinstance(val, bytes):
                val = val.decode("ascii", errors="replace")
            h_fixed[key] = val

        # parse user keywords & decode standard keywords
        h_keywords = {}

        for field in h_fixed["keywords"].split("\x00"):
            if "=" in field:
                key, value = field.split("=", 1)
                h_keywords[key] = value

        # variable (adjunct) header parsing
        if h_fixed["type"] in (1000, 1001):
            h_adjunct = {
                "xstart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xdelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xunits": struct.unpack(f"{endian}i", handle.read(4))[0],
            }
        elif h_fixed["type"] == 2000:
            h_adjunct = {
                "xstart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xdelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xunits": struct.unpack(f"{endian}i", handle.read(4))[0],
                "subsize": struct.unpack(f"{endian}i", handle.read(4))[0],
                "ystart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "ydelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "yunits": struct.unpack(f"{endian}i", handle.read(4))[0],
            }
        else:
            h_adjunct = handle.read(256)

    ver_lut = {"1.0": "BLUE 1.0", "1.1": "BLUE 1.1", "2.0": "Platinum"}
    spec_str = ver_lut.get(h_keywords.get("VER", "1.0"))
    log.info(f"Read {h_fixed['version']} type {h_fixed['type']} using {spec_str} specification.")

    validate_fixed(h_fixed)
    validate_adjunct(h_adjunct)

    return h_fixed, h_keywords, h_adjunct


def read_extended_header(file_path, h_fixed):
    """
    Read Extended Header from a BLUE file.

    Parameters
    ----------
    file_path : str
        Path to the BLUE file.
    h_fixed : dict
        Fixed Header containing 'ext_size' and 'ext_start'.
    endian : str, optional
        Endianness ('<' for little-endian, '>' for big-endian).

    Returns
    -------
    list of dict
        List of dictionaries containing parsed records.

    Raises
    ------
    SigMFConversionError
        If the extended header cannot be parsed.
    """
    entries = []
    if h_fixed["ext_size"] <= 0:
        return entries
    endian = "<" if h_fixed.get("head_rep") == "EEEI" else ">"
    with open(file_path, "rb") as handle:
        handle.seek(int(h_fixed["ext_start"]) * BLOCK_SIZE_BYTES)
        bytes_remaining = int(h_fixed["ext_size"])
        while bytes_remaining > 0:
            lkey = struct.unpack(f"{endian}i", handle.read(4))[0]
            lext = struct.unpack(f"{endian}h", handle.read(2))[0]
            ltag = struct.unpack(f"{endian}b", handle.read(1))[0]
            type_char = handle.read(1).decode("ascii", errors="replace")

            dtype, bytes_per_element = TYPE_MAP.get(type_char, (np.dtype("S1"), 1))
            val_len = lkey - lext
            val_count = val_len // bytes_per_element if bytes_per_element else 0

            if type_char == "A":
                raw = handle.read(val_len)
                if len(raw) < val_len:
                    raise SigMFConversionError("Unexpected end of extended header")
                value = raw.rstrip(b"\x00").decode("ascii", errors="replace")
            else:
                value = np.frombuffer(handle.read(val_len), dtype=dtype, count=val_count)
                if value.size == 1:
                    value = value[0]
                else:
                    value = value.tolist()

            tag = handle.read(ltag).decode("ascii", errors="replace") if ltag > 0 else ""

            total = 4 + 2 + 1 + 1 + val_len + ltag
            pad = (8 - (total % 8)) % 8
            if pad:
                handle.read(pad)

            entries.append({"tag": tag, "type": type_char, "value": value, "lkey": lkey, "lext": lext, "ltag": ltag})
            bytes_remaining -= lkey

    validate_extended_header(entries)

    return entries


def write_data(blue_path: Path, out_path: Path, h_fixed: dict) -> np.ndarray:
    """
    Write SigMF data file from BLUE file samples.

    Parameters
    ----------
    blue_path : Path
        Path to the BLUE file.
    out_path : Path
        Path to output SigMF metadata file.
    h_fixed : dict
        Header Control Block dictionary.

    Returns
    -------
    numpy.ndarray
        Parsed samples.
    """
    log.debug("parsing BLUE file data values")

    file_size_bytes = os.path.getsize(blue_path)
    extended_header_data_size = h_fixed.get("ext_size")
    fmt = h_fixed.get("format")

    # Determine destination path for SigMF data file
    dest_path = out_path.with_suffix(".sigmf-data")

    config = DATA_TYPE_CONFIGS[fmt]
    np_dtype = config["dtype"]
    is_complex = config["complex"]
    should_normalize = config["normalize"]

    # calculate element size and count
    elem_size = np.dtype(np_dtype).itemsize
    elem_count = (file_size_bytes - extended_header_data_size) // elem_size

    # read raw samples
    raw_samples = np.fromfile(blue_path, dtype=np_dtype, offset=HEADER_SIZE_BYTES, count=elem_count)

    if is_complex:
        # complex data: already in IQIQIQ... format or native complex
        if np_dtype == np.complex64:
            # already complex, no reassembly needed
            samples = raw_samples
        else:
            # reassemble interleaved IQ samples
            samples = raw_samples[::2] + 1j * raw_samples[1::2]
            # normalize if needed
            if should_normalize:
                samples = samples.astype(np.float32) / NORMALIZATION_FACTORS[fmt]
    else:
        # scalar data
        samples = raw_samples
        if should_normalize:
            samples = samples.astype(np.float32) / NORMALIZATION_FACTORS[fmt]

    # save out as SigMF IQ data file
    samples.tofile(dest_path)
    log.info("wrote %s", dest_path)

    # return the IQ data if needed for further processing if needed
    return samples


def construct_sigmf(out_path: Path, h_fixed: dict, h_keywords: dict, h_adjunct: dict, h_extended: list) -> SigMFFile:
    """
    Built & write a SigMF object from BLUE metadata.

    Parameters
    ----------
    out_path : Path
        Path to output SigMF metadata file.
    h_fixed : dict
        Fixed Header
    h_keywords : dict
        Custom User Keywords
    h_adjunct : dict
        Adjunct Header
    h_extended : list of dict
        Parsed extended header entries from read_extended_header().

    Returns
    -------
    dict
        SigMF metadata structure.
    """
    # helper to look up extended header values by tag
    def get_tag(tag):
        for entry in h_extended:
            if entry["tag"] == tag:
                return entry["value"]
        return None

    # data_rep: 'EEEI' or 'IEEE' (little or big data endianess representation)
    data_rep = h_fixed.get("data_rep")

    # data_format: for example 'CI' or 'SD' (data format code - real or complex, int or float)
    data_format = h_fixed.get("format")
    endian_suffix = "_le" if data_rep == "EEEI" else "_be"

    # get base datatype and add endianness
    base_datatype = DATATYPE_MAP_BASE.get(data_format)
    datatype = base_datatype + endian_suffix

    log.info(f"Using SigMF datatype: {datatype} for BLUE format {h_fixed['format']} and endianness {data_rep}.")

    # sample rate: prefer adjunct.xdelta, else extended header SAMPLE_RATE
    if "xdelta" in h_adjunct:
        sample_rate_hz = 1 / h_adjunct["xdelta"]
    else:
        sample_rate_hz = float(get_tag("SAMPLE_RATE"))

    if "outlets" in h_fixed and h_fixed["outlets"] > 0:
        num_channels = int(h_fixed["outlets"])
    else:
        num_channels = 1

    # base global metadata
    global_info = {
        # FIXME: what common fields are in h_fixed?
        "core:author": getpass.getuser(),
        SigMFFile.DATATYPE_KEY: datatype,
        # FIXME: what is the most apt description?
        # SigMFFile.DESCRIPTION_KEY: ???,
        SigMFFile.RECORDER_KEY: "Official SigMF BLUE converter",
        SigMFFile.NUM_CHANNELS_KEY: num_channels,
        SigMFFile.SAMPLE_RATE_KEY: sample_rate_hz,
        SigMFFile.EXTENSIONS_KEY: [{"name": "blue", "version": "0.0.1", "optional": True}],
    }

    # merge HCB values into metadata
    global_info["blue:fixed"] = h_fixed
    global_info["blue:keywords"] = h_keywords
    global_info["blue:adjunct"] = h_adjunct

    # merge extended header fields
    if h_extended:
        extended = {}
        for entry in h_extended:
            key = entry.get("tag")
            value = entry.get("value")
            if hasattr(value, "item"):
                value = value.item()
            extended[key] = value
        global_info["blue:extended"] = extended

    # BLUE uses 1950-01-01 as epoch, UNIX uses 1970-01-01
    blue_start_time = float(h_fixed.get("timecode", 0))
    blue_start_time += h_adjunct.get("xstart", 0)
    blue_start_time += float(h_keywords.get("TC_PREC", 0))

    blue_epoch = blue_start_time - 631152000  # seconds between 1950 and 1970
    # FIXME: I am unsure if the timezone is always UTC in these files
    blue_datetime = datetime.fromtimestamp(blue_epoch, tz=timezone.utc)

    capture_info = {
        SigMFFile.DATETIME_KEY: blue_datetime.strftime(SIGMF_DATETIME_ISO8601_FMT),
    }

    if get_tag("RF_FREQ") is not None:
        # FIXME: I believe there are many possible keys related to tune frequency
        capture_info[SigMFFile.FREQUENCY_KEY] = float(get_tag("RF_FREQ"))

    # actually write to SigMF
    filenames = get_sigmf_filenames(out_path)

    meta = SigMFFile(
        data_file=filenames["data_fn"],
        global_info=global_info,
    )
    meta.add_capture(0, metadata=capture_info)
    log.debug("created %r", meta)

    meta.tofile(filenames["meta_fn"], toarchive=False)
    log.info("wrote %s", filenames["meta_fn"])

    return meta


def validate_file(blue_path: Path) -> None:
    """
    Basic validation of the BLUE file.

    Parameters
    ----------
    blue_path : Path
        Path to the BLUE file.

    Raises
    ------
    SigMFConversionError
        If the file is abnormal.
    """
    if blue_path.stat().st_size < 512:
        raise SigMFConversionError("BLUE file is too small to contain required headers.")


def validate_fixed(h_fixed: dict) -> None:
    """
    Check that Fixed Header contains minimum required fields.

    Parameters
    ----------
    h_fixed : dict
        Fixed Header dictionary.

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    required = ["version", "data_start", "data_size", "data_rep", "head_rep", "detached", "format", "type"]
    for field in required:
        if field not in h_fixed:
            raise SigMFConversionError(f"Missing required Fixed Header field: {field}")
        # FIXME: when could this possibly occur?
        if h_fixed[field] is None:
            raise SigMFConversionError(f"Required Fixed Header field {field} is None")

    for rep_field in ["data_rep", "head_rep"]:
        if h_fixed[rep_field] not in ("EEEI", "IEEE"):
            raise SigMFConversionError(f"Invalid value for {rep_field}: {h_fixed[rep_field]}")
    # FIXME: merge these lookup tables into one
    if h_fixed["format"] not in DATATYPE_MAP_BASE:
        raise SigMFConversionError(f"Unsupported data format: {h_fixed['format']}")
    if h_fixed["format"] not in DATA_TYPE_CONFIGS:
        raise SigMFConversionError(f"Unsupported data format: {h_fixed['format']}")


def validate_adjunct(adjunct: dict) -> None:
    """
    Check that the Adjunct header contains minimum required fields.
    """
    # validate xdelta (1 / samp_rate) if present
    if "xdelta" in adjunct:
        xdelta = adjunct["xdelta"]
        if xdelta <= 0:
            raise SigMFConversionError(f"Invalid adjunct xdelta time interval: {xdelta}")


def validate_extended_header(entries: list) -> None:
    """
    Check that BLUE Extended Header contains minimum required fields.

    Parameters
    ----------
    entries : list of dict
        List of extended header entries.

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    # check for SAMPLE_RATE if present
    for entry in entries:
        if entry["tag"] == "SAMPLE_RATE":
            sample_rate = float(entry["value"])
            if sample_rate <= 0:
                raise SigMFConversionError(f"Invalid SAMPLE_RATE in extended header: {sample_rate}")


def convert_blue(
    blue_path: str,
    out_path: Optional[str] = None,
) -> np.ndarray:
    """
    Convert a MIDIS Bluefile to SigMF metadata and data.

    Parameters
    ----------
    blue_path : str
        Path to the Blue file.
    out_path : str
        Path to the output SigMF metadata file.

    Returns
    -------
    numpy.ndarray
        IQ Data.

    Notes
    -----
    This function currently reads BLUE then writes a SigMF pair. We could also
    implement a function that instead writes metadata only for a non-conforming
    dataset using the HEADER_BYTES_KEY and TRAILING_BYTES_KEY in most cases.
    """
    blue_path = Path(blue_path)
    if out_path is None:
        # extension will be changed later
        out_path = Path(blue_path)
    else:
        out_path = Path(out_path)

    validate_file(blue_path)

    # read Header control block (HCB) to determine how to process the rest of the file
    h_fixed, h_keywords, h_adjunct = read_hcb(blue_path)

    # read extended header
    h_extended = read_extended_header(blue_path, h_fixed)

    # write to SigMF data file
    _ = write_data(blue_path, out_path, h_fixed)

    log.debug(">>>>>>>>> Fixed Header")
    for key, _, _, _, desc in FIXED_LAYOUT:
        log.debug(f"{key:10s}: {h_fixed[key]!r}  # {desc}")

    log.debug(">>>>>>>>> User Keywords")
    log.debug(h_keywords)

    log.debug(">>>>>>>>> Adjunct Header")
    log.debug(h_adjunct)

    log.debug(">>>>>>>>> Extended Header")
    for entry in h_extended:
        log.debug(f"{entry['tag']:20s}:{entry['value']}")

    # call the SigMF conversion for metadata generation
    meta = construct_sigmf(out_path, h_fixed, h_keywords, h_adjunct, h_extended)

    return meta


def main() -> None:
    """
    Entry-point for sigmf_convert_blue
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str, help="Blue (cdif) file path")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"%(prog)s v{toolversion}")
    args = parser.parse_args()

    level_lut = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    logging.basicConfig(level=level_lut[min(args.verbose, 2)])

    convert_blue(args.input)

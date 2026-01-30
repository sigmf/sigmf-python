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

import base64
import getpass
import io
import logging
import struct
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from packaging.version import InvalidVersion, Version

from .. import __version__ as toolversion
from ..error import SigMFConversionError
from ..sigmffile import SigMFFile, fromfile, get_sigmf_filenames
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

BLOCK_SIZE_BYTES = 512

TYPE_MAP = {
    # BLUE format code to numpy dtype
    # note: new non 1-1 mapping supported needs new handling in data_loopback
    # "P" : packed bits,
    "A": np.dtype("S1"),  # ASCII for unpacking text fields
    # "N" : 4-bit integer,
    "B": np.int8,
    "U": np.uint16,
    "I": np.int16,
    "V": np.uint32,
    "L": np.int32,
    "F": np.float32,
    "X": np.int64,
    "D": np.float64,
    # "O": excess-128,
}


def blue_to_sigmf_type_str(h_fixed: dict) -> str:
    """
    Convert BLUE format code to SigMF datatype string.

    Parameters
    ----------
    h_fixed : dict
        Fixed Header dictionary containing 'format' and 'data_rep' fields.

    Returns
    -------
    str
        SigMF datatype string (e.g., 'ci16_le', 'rf32_be').
    """
    # extract format code and endianness from header
    format_code = h_fixed.get("format")
    endianness = h_fixed.get("data_rep")

    # parse format code components
    is_complex = format_code[0] == "C"
    numpy_dtype = TYPE_MAP[format_code[1]]

    # compute everything from numpy dtype
    dtype_obj = np.dtype(numpy_dtype)
    bits = dtype_obj.itemsize * 8  # bytes to bits

    # infer sigmf type from numpy kind
    if dtype_obj.kind == "u":
        sigmf_type = "u"
    elif dtype_obj.kind == "i":
        sigmf_type = "i"
    else:
        sigmf_type = "f"

    # build datatype string
    prefix = "c" if is_complex else "r"
    datatype = f"{prefix}{sigmf_type}{bits}"

    # add endianness for types > 8 bits
    if bits > 8:
        endian_suffix = "_le" if endianness == "EEEI" else "_be"
        datatype += endian_suffix

    return datatype


def detect_endian(data: bytes) -> str:
    """
    Detect endianness of a Bluefile header.

    Parameters
    ----------
    data : bytes
        Raw header data.

    Returns
    -------
    str
        "<" for little-endian or ">" for big-endian.

    Raises
    ------
    SigMFConversionError
        If the endianness is unexpected.
    """
    endianness = data[8:12].decode("ascii")
    if endianness == "EEEI":
        return "<"
    if endianness == "IEEE":
        return ">"
    raise SigMFConversionError(f"Unsupported endianness: {endianness}")


def read_hcb(file_path: Path) -> (dict, dict):
    """
    Read Header Control Block (HCB) from BLUE file.

    First 256 bytes contains fixed header, followed by 256 bytes of adjunct header.

    Parameters
    ----------
    file_path : Path
        Path to the Blue file.

    Returns
    -------
    h_fixed : dict
        Fixed Header
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
            except struct.error as err:
                raise SigMFConversionError(f"Failed to unpack field {key} with endian {endian}") from err
            if isinstance(val, bytes):
                val = val.decode("ascii", errors="replace")
            h_fixed[key] = val

        # parse user keywords & decode standard keywords
        h_keywords = {}

        for field in h_fixed["keywords"].split("\x00"):
            if "=" in field:
                key, value = field.split("=", 1)
                h_keywords[key] = value
        # place parsed keywords back into fixed header
        h_fixed["keywords"] = h_keywords

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
            # read raw adjunct header as bytes and convert to base64 for JSON serialization
            log.warning(f"Unknown BLUE file type {h_fixed['type']}, encoding adjunct header in metadata as base64.")
            raw_adjunct = handle.read(256)
            h_adjunct = {"raw_base64": base64.b64encode(raw_adjunct).decode("ascii")}

        if h_fixed["keywords"].get("CRC") is not None:
            # CRC calculated on first 160 bytes of fixed header + full extended header
            handle.seek(0)
            buffer = handle.read(160)
            handle.seek(int(h_fixed["ext_start"]) * BLOCK_SIZE_BYTES)
            buffer += handle.read(int(h_fixed["ext_size"]))
            target_crc = h_fixed["keywords"]["CRC"].lower()
            if target_crc == _crc32_broken(buffer):
                log.debug("CRC ok (BLUE implementation)")
            elif target_crc == _crc32_posix(buffer):
                log.debug("CRC ok (POSIX implementation)")
            else:
                log.warning("CRC mismatch in BLUE metadata!")

    validate_fixed(h_fixed)
    validate_adjunct(h_adjunct)

    return h_fixed, h_adjunct


def read_extended_header(file_path: Path, h_fixed: dict) -> list:
    """
    Read Extended Header from a BLUE file.

    Parameters
    ----------
    file_path : str
        Path to the BLUE file.
    h_fixed : dict
        Fixed Header containing 'ext_size' and 'ext_start'.

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

            # get dtype and compute bytes per element
            if type_char in TYPE_MAP:
                dtype = TYPE_MAP[type_char]
                bytes_per_element = np.dtype(dtype).itemsize
            else:
                # fallback for unknown types
                dtype = np.dtype("S1")
                bytes_per_element = 1

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
                    val_item = value[0]
                    # handle bytes first (numpy.bytes_ is also np.generic)
                    if isinstance(val_item, bytes):
                        # handle bytes from S1 dtype - convert to base64 for JSON
                        value = base64.b64encode(val_item).decode("ascii")
                    elif isinstance(val_item, np.generic):
                        # convert numpy scalar to native python type
                        value = val_item.item()
                    else:
                        value = val_item
                else:
                    value = value.tolist()

            tag = handle.read(ltag).decode("ascii", errors="replace") if ltag > 0 else ""

            total = 4 + 2 + 1 + 1 + val_len + ltag
            pad = (8 - (total % 8)) % 8
            if pad:
                handle.read(pad)

            entries.append({"tag": tag, "type": type_char, "value": value, "lkey": lkey, "lext": lext, "ltag": ltag})
            bytes_remaining -= lkey

    validate_extended(entries)

    return entries


def data_loopback(blue_path: Path, data_path: Path, h_fixed: dict) -> None:
    """
    Write SigMF data file from BLUE file samples.

    Parameters
    ----------
    blue_path : Path
        Path to the BLUE file.
    data_path : Path
        Destination path for the SigMF dataset (.sigmf-data).
    h_fixed : dict
        Header Control Block dictionary.
    """
    header_bytes, data_bytes, _ = _get_blue_boundaries(blue_path, h_fixed)

    # check for zero-sample file (metadata-only)
    if data_bytes == 0:
        log.info("detected zero-sample BLUE file, creating metadata-only SigMF")
        return

    # read raw samples
    with open(blue_path, "rb") as handle:
        handle.seek(header_bytes)
        raw_data = handle.read(data_bytes)

    with open(data_path, "wb") as handle:
        handle.write(raw_data)
    log.info("wrote SigMF dataset to %s", data_path)


@lru_cache()
def _generate_crc_table(poly: int):
    """generate lookup table for given polynomial"""
    table = []
    for idx in range(256):
        crc = idx << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = (crc << 1) ^ poly
            else:
                crc = crc << 1
            crc &= 0xFFFFFFFF
        table.append(crc)
    return table


def _crc32_posix(data: bytes) -> str:
    """
    POSIX.2 CRC-32 with buffer length included

    Supposed BLUE standard implementation.

    Test Vector
    -----------
    >>> _crc32_posix(bytes.fromhex('deadbeef'))
    '1c3cd7e6'
    """
    table = _generate_crc_table(0x04C11DB7)
    crc = 0

    # process data bytes
    for byte_val in data:
        crc = (crc << 8) ^ table[(crc >> 24) ^ byte_val]
        crc &= 0xFFFFFFFF

    # include buffer size in calculation
    size_val = len(data)
    while size_val > 0:
        crc = (crc << 8) ^ table[(crc >> 24) ^ (size_val & 0xFF)]
        crc &= 0xFFFFFFFF
        size_val >>= 8

    return f"{(~crc) & 0xFFFFFFFF:08x}"


def _crc32_broken(data: bytes) -> str:
    """
    Similar to posix but with a broken length calculation.

    Used in many BLUE files.

    Test Vector
    -----------
    >>> _crc32_broken(bytes.fromhex('deadbeef'))
    '48281aa6'
    """
    table = _generate_crc_table(0x04C11DB7)
    crc = 0xFFFFFFFF

    # process data bytes
    for byte_val in data:
        crc = (crc << 8) ^ table[(crc >> 24) ^ byte_val]
        crc &= 0xFFFFFFFF

    # broken length calculation - only processes high byte of length
    size_val = len(data)
    while size_val > 0:
        crc = (crc << 8) ^ table[(crc >> 24) ^ (size_val >> 24)]
        crc &= 0xFFFFFFFF
        size_val >>= 8

    return f"{(~crc) & 0xFFFFFFFF:08x}"


def _get_blue_boundaries(blue_path: Path, h_fixed: dict) -> (int, int):
    """
    Extract data boundaries from fixed header.
    """
    file_bytes = blue_path.stat().st_size
    header_bytes = int(h_fixed.get("data_start"))
    data_bytes = int(h_fixed.get("data_size"))
    trailing_bytes = file_bytes - (header_bytes + data_bytes)
    return header_bytes, data_bytes, trailing_bytes


def _description(h_fixed: dict) -> str:
    """
    Construct a human-readable description of the BLUE file.
    """
    try:
        spec_str = "Unknown"
        version = Version(h_fixed.get("keywords").get("VER", "0.0"))
        if version.major == 1:
            spec_str = f"BLUE {version}"
        elif version.major == 2:
            spec_str = f"Platinum {version}"
    except InvalidVersion:
        log.warning("Could not parse BLUE specification from VER keyword.")
    # h_fixed will contain number e.g. 1000, 1001, 2000, 2001
    description = (
        f"Read {h_fixed['version']} type {h_fixed['type']} {h_fixed['format']} using {spec_str} specification."
    )
    log.info(description)
    return description


def _build_common_metadata(
    h_fixed: dict,
    h_adjunct: dict,
    h_extended: list,
    is_ncd: bool = False,
    blue_file_name: str = None,
    trailing_bytes: int = 0,
) -> Tuple[dict, dict]:
    """
    Build common global_info and capture_info metadata for both standard and NCD SigMF files.

    Parameters
    ----------
    h_fixed : dict
        Fixed Header
    h_adjunct : dict
        Adjunct Header
    h_extended : list of dict
        Parsed extended header entries.
    is_ncd : bool, optional
        If True, adds NCD-specific fields.
    blue_file_name : str, optional
        Original BLUE file name (for NCD).
    trailing_bytes : int, optional
        Number of trailing bytes (for NCD).

    Returns
    -------
    tuple[dict, dict]
        (global_info, capture_info) dictionaries.
    """
    # helper to look up extended header values by tag
    def get_tag(tag):
        for entry in h_extended:
            if entry["tag"] == tag:
                return entry["value"]
        return None

    # get sigmf datatype from blue format and endianness
    datatype = blue_to_sigmf_type_str(h_fixed)

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
        SigMFFile.AUTHOR_KEY: getpass.getuser(),
        SigMFFile.DATATYPE_KEY: datatype,
        SigMFFile.RECORDER_KEY: "Official SigMF BLUE converter",
        SigMFFile.NUM_CHANNELS_KEY: num_channels,
        SigMFFile.SAMPLE_RATE_KEY: sample_rate_hz,
        SigMFFile.EXTENSIONS_KEY: [{"name": "blue", "version": "0.0.1", "optional": True}],
        SigMFFile.DESCRIPTION_KEY: _description(h_fixed),
    }

    # add NCD-specific fields
    if is_ncd:
        global_info[SigMFFile.DATASET_KEY] = blue_file_name
        global_info[SigMFFile.TRAILING_BYTES_KEY] = trailing_bytes

    # merge HCB values into metadata
    global_info["blue:fixed"] = h_fixed
    global_info["blue:adjunct"] = h_adjunct

    # merge extended header fields, handling duplicate keys
    if h_extended:
        extended = {}
        tag_counts = {}
        for entry in h_extended:
            tag = entry.get("tag")
            value = entry.get("value")
            if hasattr(value, "item"):
                value = value.item()

            # handle duplicate tags by numbering them
            if tag in extended:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                numbered_tag = f"{tag}_{tag_counts[tag]}"
                extended[numbered_tag] = value
            else:
                extended[tag] = value
        global_info["blue:extended"] = extended

    # calculate blue start time
    blue_start_time = float(h_fixed.get("timecode", 0))
    blue_start_time += h_adjunct.get("xstart", 0)
    blue_start_time += float(h_fixed.get("keywords").get("TC_PREC", 0))

    capture_info = {}

    if blue_start_time == 0:
        log.warning("BLUE timecode is zero or missing; capture datetime metadata will be absent.")
    else:
        # timecode uses 1950-01-01 as epoch, datetime uses 1970-01-01
        blue_epoch = blue_start_time - 631152000  # seconds between 1950 and 1970
        blue_datetime = datetime.fromtimestamp(blue_epoch, tz=timezone.utc)
        capture_info[SigMFFile.DATETIME_KEY] = blue_datetime.strftime(SIGMF_DATETIME_ISO8601_FMT)

    if get_tag("RF_FREQ") is not None:
        # it's possible other keys indicate tune frequency, but RF_FREQ is common
        capture_info[SigMFFile.FREQUENCY_KEY] = float(get_tag("RF_FREQ"))

    return global_info, capture_info


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
    Check that Fixed Header contains minimum required fields for parsing.

    Parameters
    ----------
    h_fixed : dict
        Fixed Header dictionary.

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    required = ["version", "data_start", "data_size", "data_rep", "head_rep", "detached", "format", "type", "keywords"]
    for field in required:
        if field not in h_fixed:
            raise SigMFConversionError(f"Missing required Fixed Header field: {field}")
    for rep_field in ["data_rep", "head_rep"]:
        if h_fixed[rep_field] not in ("EEEI", "IEEE"):
            raise SigMFConversionError(f"Invalid value for {rep_field}: {h_fixed[rep_field]}")
    if h_fixed["data_size"] < 0:
        raise SigMFConversionError(f"Invalid data_size: {h_fixed['data_size']} (must be >= 0)")
    if len(h_fixed["format"]) != 2 or h_fixed["format"][0] not in "SC" or h_fixed["format"][1] not in TYPE_MAP:
        raise SigMFConversionError(f"Unsupported data format: {h_fixed['format']}")


def validate_adjunct(h_adjunct: dict) -> None:
    """
    Check that the Adjunct header contains minimum required fields.

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    # validate xdelta (1 / samp_rate) if present
    if "xdelta" in h_adjunct:
        xdelta = h_adjunct["xdelta"]
        if xdelta <= 0:
            raise SigMFConversionError(f"Invalid adjunct xdelta time interval: {xdelta}")


def validate_extended(entries: list) -> None:
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


def construct_sigmf(
    filenames: dict,
    h_fixed: dict,
    h_adjunct: dict,
    h_extended: list,
    is_metadata_only: bool = False,
    create_archive: bool = False,
) -> SigMFFile:
    """
    Built & write a SigMF object from BLUE metadata.

    Parameters
    ----------
    filenames : dict
        Mapping returned by get_sigmf_filenames containing destination paths.
    h_fixed : dict
        Fixed Header
    h_adjunct : dict
        Adjunct Header
    h_extended : list of dict
        Parsed extended header entries from read_extended_header().
    is_metadata_only : bool, optional
        If True, creates a metadata-only SigMF file.
    create_archive : bool, optional
        When True, package output as SigMF archive instead of a meta/data pair.

    Returns
    -------
    SigMFFile
        SigMF object.
    """
    # use shared helper to build common metadata
    global_info, capture_info = _build_common_metadata(h_fixed, h_adjunct, h_extended)

    # set metadata-only flag for zero-sample files (only for non-NCD files)
    if is_metadata_only:
        # ensure we're not accidentally setting metadata_only for an NCD
        if SigMFFile.DATASET_KEY in global_info:
            raise ValueError(
                "Cannot set metadata_only=True for Non-Conforming Dataset files. "
                "Per SigMF spec, metadata_only MAY NOT be used with core:dataset field."
            )
        global_info[SigMFFile.METADATA_ONLY_KEY] = True

    # for metadata-only files, don't specify data_file and skip checksum
    if is_metadata_only:
        meta = SigMFFile(
            data_file=None,
            global_info=global_info,
            skip_checksum=True,
        )
        meta.data_buffer = io.BytesIO()
    else:
        meta = SigMFFile(
            data_file=filenames["data_fn"],
            global_info=global_info,
        )
    meta.add_capture(0, metadata=capture_info)

    if create_archive:
        meta.tofile(filenames["archive_fn"], toarchive=True)
        log.info("wrote SigMF archive to %s", filenames["archive_fn"])
        # metadata returned should be for this archive
        meta = fromfile(filenames["archive_fn"])
    else:
        meta.tofile(filenames["meta_fn"], toarchive=False)
        log.info("wrote SigMF metadata to %s", filenames["meta_fn"])

    log.debug("created %r", meta)
    return meta


def construct_sigmf_ncd(
    blue_path: Path,
    h_fixed: dict,
    h_adjunct: dict,
    h_extended: list,
) -> SigMFFile:
    """
    Construct Non-Conforming Dataset SigMF metadata for BLUE file.

    Parameters
    ----------
    blue_path : Path
        Path to the original BLUE file.
    h_fixed : dict
        Fixed Header
    h_adjunct : dict
        Adjunct Header
    h_extended : list of dict
        Parsed extended header entries from read_extended_header().

    Returns
    -------
    SigMFFile
        NCD SigMF object pointing to original BLUE file.
    """
    header_bytes, data_bytes, trailing_bytes = _get_blue_boundaries(blue_path, h_fixed)

    # use shared helper to build common metadata, with NCD-specific additions
    global_info, capture_info = _build_common_metadata(
        h_fixed,
        h_adjunct,
        h_extended,
        is_ncd=True,
        blue_file_name=blue_path.name,
        trailing_bytes=trailing_bytes,
    )

    # add NCD-specific capture info
    capture_info[SigMFFile.HEADER_BYTES_KEY] = header_bytes

    # create NCD metadata-only SigMF pointing to original file
    meta = SigMFFile(global_info=global_info, skip_checksum=True)
    meta.set_data_file(data_file=blue_path, offset=header_bytes, skip_checksum=True, size_bytes=data_bytes)
    meta.data_buffer = io.BytesIO()
    meta.add_capture(0, metadata=capture_info)
    log.debug("created NCD SigMF: %r", meta)

    return meta


def blue_to_sigmf(
    blue_path: str,
    out_path: Optional[str] = None,
    create_archive: bool = False,
    create_ncd: bool = False,
) -> SigMFFile:
    """
    Read a MIDAS Bluefile, optionally write SigMF, return associated SigMF object.

    Parameters
    ----------
    blue_path : str
        Path to the Blue file.
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
    """
    log.debug(f"read {blue_path}")

    # auto-enable NCD when no output path is specified
    if out_path is None:
        create_ncd = True

    blue_path = Path(blue_path)
    if out_path is None:
        base_path = blue_path
    else:
        base_path = Path(out_path)

    filenames = get_sigmf_filenames(base_path)

    # ensure output directory exists
    filenames["base_fn"].parent.mkdir(parents=True, exist_ok=True)

    validate_file(blue_path)

    # read Header control block (HCB) to determine how to process the rest of the file
    h_fixed, h_adjunct = read_hcb(blue_path)

    # read extended header
    h_extended = read_extended_header(blue_path, h_fixed)

    # check if this is a zero-sample (metadata-only) file
    data_size_bytes = int(h_fixed.get("data_size", 0))
    metadata_only = data_size_bytes == 0

    # handle NCD case
    if create_ncd:
        # create metadata-only SigMF for NCD pointing to original file
        ncd_meta = construct_sigmf_ncd(blue_path, h_fixed, h_adjunct, h_extended)

        # write NCD metadata to specified output path if provided
        if out_path is not None:
            ncd_meta.tofile(filenames["meta_fn"])
            log.info("wrote SigMF non-conforming metadata to %s", filenames["meta_fn"])

        return ncd_meta

    with tempfile.TemporaryDirectory() as temp_dir:
        if not metadata_only:
            if create_archive:
                # for archives, write data to a temporary file that will be cleaned up
                data_path = Path(temp_dir) / filenames["data_fn"].name
                filenames["data_fn"] = data_path  # update path for construct_sigmf
            else:
                # for file pairs, write to the final destination
                data_path = filenames["data_fn"]
            data_loopback(blue_path, data_path, h_fixed)
        else:
            log.info("skipping data file creation for zero-sample BLUE file")

        # call the SigMF conversion for metadata generation
        meta = construct_sigmf(
            filenames=filenames,
            h_fixed=h_fixed,
            h_adjunct=h_adjunct,
            h_extended=h_extended,
            is_metadata_only=metadata_only,
            create_archive=create_archive,
        )

    log.debug(">>>>>>>>> Fixed Header")
    for key, _, _, _, desc in FIXED_LAYOUT:
        log.debug(f"{key:10s}: {h_fixed[key]!r}  # {desc}")

    log.debug(">>>>>>>>> Adjunct Header")
    log.debug(h_adjunct)

    log.debug(">>>>>>>>> Extended Header")
    for entry in h_extended:
        log.debug(f"{entry['tag']:20s}:{entry['value']}")

    return meta

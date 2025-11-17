# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Blue File converter with HCB and Extended Header Parsing
This script reads and parses the HCB (Header Control Block) and extended header keywords from a Blue file format.
It supports different file types and extracts metadata accordingly.
Converts the extracted metadata into SigMF format.
"""

import argparse
import hashlib
import json
import logging
import os
import struct
from datetime import datetime, timezone

import numpy as np

from .. import __version__ as toolversion
from ..error import SigMFConversionError

log = logging.getLogger()

# fmt: off
HCB_LAYOUT = [
    # HCB field definitions: (name, offset, size, fmt, description) up to adjunct
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
    ("format",    52,  2,  "2s",   "Data format code"),
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
    # Adjunct starts at 256
]
# fmt: on

TYPE_MAP = {
    # Extended header type map
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


def detect_endian(data, layout, probe_fields=("data_size", "version")):
    """
    Detect endianness of a Bluefile header.

    TODO: Look at this code and see if can be improved and possibly simplified.

    Parameters
    ----------
    data : bytes
        Raw header data.
    layout : list of tuples
        HCB layout definition (name, offset, size, fmt, desc).
    probe_fields : tuple of str, optional
        Field names to test for sanity checks.

    Returns
    -------
    str
        "<" for little-endian or ">" for big-endian.
    """
    # TODO: handle both types of endianess 'EEEI' or IEEE and data rep and signal rep
    endianness = data[8:12].decode("utf-8")
    log.debug(f"endianness: {endianness}")
    if endianness not in ("EEEI", "IEEE"):
        raise SigMFConversionError(f"Unexpected endianness: {endianness}")

    for endian in ("<", ">"):
        ok = True
        for name, offset, size, fmt, desc in layout:
            if name not in probe_fields:
                continue
            raw = data[offset : offset + size]
            try:
                val = struct.unpack(endian + fmt, raw)[0]
                # sanity checks
                MAX_DATA_SIZE_FACTOR = 100

                if name == "data_size":
                    if val <= 0 or val > len(data) * MAX_DATA_SIZE_FACTOR:
                        ok = False
                        break
                elif name == "version":
                    if not (0 < val < 10):  # expect small version number
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
    Read HCB fields and adjunct block from a Blue file.

    Parameters
    ----------
    file_path : str
        Path to the Blue file.

    Returns
    -------
    dict
        Parsed HCB fields and adjunct metadata.
    """

    hcb = {}
    with open(file_path, "rb") as handle:
        data = handle.read(HEADER_SIZE_BYTES)
        endian = detect_endian(data, HCB_LAYOUT)

        # fixed fields
        for name, offset, size, fmt, desc in HCB_LAYOUT:
            raw = data[offset : offset + size]
            try:
                val = struct.unpack(endian + fmt, raw)[0]
            except struct.error:
                raise ValueError(f"Failed to unpack field {name} with endian {endian}")
            if isinstance(val, bytes):
                val = val.decode("ascii", errors="replace").strip("\x00 ")
            hcb[name] = val

        # adjunct parsing
        adjunct_offset_bytes = 256
        handle.seek(adjunct_offset_bytes)
        if hcb["type"] in (1000, 1001):
            hcb["adjunct"] = {
                "xstart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xdelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xunits": struct.unpack(f"{endian}i", handle.read(4))[0],
            }
        elif hcb["type"] == 2000:
            hcb["adjunct"] = {
                "xstart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xdelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "xunits": struct.unpack(f"{endian}i", handle.read(4))[0],
                "subsize": struct.unpack(f"{endian}i", handle.read(4))[0],
                "ystart": struct.unpack(f"{endian}d", handle.read(8))[0],
                "ydelta": struct.unpack(f"{endian}d", handle.read(8))[0],
                "yunits": struct.unpack(f"{endian}i", handle.read(4))[0],
            }
        else:
            hcb["adjunct_raw"] = handle.read(adjunct_offset_bytes)

    return hcb


def parse_extended_header(file_path, hcb, endian="<"):
    """
    Parse extended header keyword records.

    Parameters
    ----------
    file_path : str
        Path to the Bluefile.
    hcb : dict
        Header Control Block containing 'ext_size' and 'ext_start'.
    endian : str, optional
        Endianness ('<' for little-endian, '>' for big-endian).

    Returns
    -------
    list of dict
        List of dictionaries containing parsed records.
    """
    if hcb["ext_size"] <= 0:
        return []
    entries = []
    with open(file_path, "rb") as handle:
        handle.seek(int(hcb["ext_start"]) * BLOCK_SIZE_BYTES)
        bytes_remaining = int(hcb["ext_size"])
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
                    raise ValueError("Unexpected end of extended header")
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

    return entries


def parse_data_values(file_path, hcb, endianness):
    """
    Parse key HCB values used for further processing.

    Parameters
    ----------
    file_path : str
        Path to the Blue file.
    hcb : dict
        Header Control Block dictionary.
    endianness : str
        Endianness ('<' for little-endian, '>' for big-endian).

    Returns
    -------
    numpy.ndarray
        Parsed samples.
    """
    log.info("parsing blue file data values")
    with open(file_path, "rb") as handle:
        data = handle.read(HEADER_SIZE_BYTES)
        if len(data) < HEADER_SIZE_BYTES:
            raise ValueError("Incomplete header")
        dtype = data[52:54].decode("utf-8")  # eg 'CI', 'CF', 'SD'
        log.debug(f"data type: {dtype}")


        time_interval = np.frombuffer(data[264:272], dtype=np.float64)[0]
        if time_interval <= 0:
            raise ValueError(f"Invalid time interval: {time_interval}")
        sample_rate_hz = 1 / time_interval
        log.info(f"sample rate: {sample_rate_hz/1e6:.3f} MHz")
        extended_header_data_size = int.from_bytes(data[28:32], byteorder="little")
        file_size_bytes = os.path.getsize(file_path)
        log.debug(f"file size: {file_size_bytes} bytes")

    # Determine destination path for SigMF data file
    dest_path = file_path.rsplit(".", 1)[0]

    ### complex data parsing

    # complex 16-bit integer  IQ data > ci16_le in SigMF
    if dtype == "CI":
        elem_size = np.dtype(np.int16).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        raw_samples = np.fromfile(file_path, dtype=np.int16, offset=HEADER_SIZE_BYTES, count=elem_count)
        # reassemble interleaved IQ samples
        samples = raw_samples[::2] + 1j * raw_samples[1::2]  # convert to IQIQIQ...
        # normalize samples to -1.0 to +1.0 range
        samples = samples.astype(np.float32) / 32767.0
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # complex 32-bit integer  IQ data > ci32_le in SigMF
    elif dtype == "CL":
        elem_size = np.dtype(np.int32).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        raw_samples = np.fromfile(file_path, dtype=np.int32, offset=HEADER_SIZE_BYTES, count=elem_count)
        # reassemble interleaved IQ samples
        samples = raw_samples[::2] + 1j * raw_samples[1::2]  # convert to IQIQIQ...
        # normalize samples to -1.0 to +1.0 range
        samples = samples.astype(np.float32) / 2147483647.0
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # complex 32-bit float  IQ data > cf32_le in SigMF
    elif dtype == "CF":
        # each complex sample is 8 bytes total (2 × float32), so np.complex64 is appropriate
        # no need to reassemble IQ — already complex
        elem_size = np.dtype(np.complex64).itemsize  # will be 8 bytes
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.complex64, offset=HEADER_SIZE_BYTES, count=elem_count)
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    ### scalar data parsing

    # scalar data parsing > ri8_le in SigMF
    elif dtype == "SB":
        elem_size = np.dtype(np.int8).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.int8, offset=HEADER_SIZE_BYTES, count=elem_count)
        # normalize samples to -1.0 to +1.0 range
        samples = samples.astype(np.float32) / 127.0
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # scalar data parsing > ri16_le in SigMF
    elif dtype == "SI":
        elem_size = np.dtype(np.int16).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.int16, offset=HEADER_SIZE_BYTES, count=elem_count)
        # normalize samples to -1.0 to +1.0 range
        samples = samples / 32767.0
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # scalar data parsing > ri32_le in SigMF
    elif dtype == "SL":
        elem_size = np.dtype(np.int32).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.int32, offset=HEADER_SIZE_BYTES, count=elem_count)
        # normalize samples to -1.0 to +1.0 range
        samples = samples / 2147483647.0
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # scalar data parsing > ri64_le in SigMF
    elif dtype == "SX":
        elem_size = np.dtype(np.int64).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.int64, offset=HEADER_SIZE_BYTES, count=elem_count)
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # scalar data parsing > rf32_le in SigMF
    elif dtype == "SF":
        elem_size = np.dtype(np.float32).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.float32, offset=HEADER_SIZE_BYTES, count=elem_count)
        # save out as SigMF IQ data file
        samples.tofile(f"{dest_path}.sigmf-data")

    # scalar data parsing > rf64_le in SigMF
    elif dtype == "SD":
        elem_size = np.dtype(np.float64).itemsize
        elem_count = (file_size_bytes - extended_header_data_size) // elem_size
        samples = np.fromfile(file_path, dtype=np.float64, offset=HEADER_SIZE_BYTES, count=elem_count)
        # save out as SigMF IQ data file
        samples.astype(np.complex64).tofile(f"{dest_path}.sigmf-data")
    else:
        raise ValueError(f"Unsupported data type: {dtype}")

    # TODO: validate handling of scalar types - reshape per mathlab port shown here?
    # return the IQ data if needed for further processing if needed
    return samples


def blue_to_sigmf(hcb, ext_entries, file_path):
    """
    Build a SigMF metadata dict from parsed Bluefile HCB and extended header.

    Parameters
    ----------
    hcb : dict
        Header Control Block from read_hcb().
    ext_entries : list of dict
        Parsed extended header entries from parse_extended_header().
    file_path : str
        Path to the original blue file.

    Returns
    -------
    dict
        SigMF metadata structure.

    Raises
    ------
    ValueError
        If required fields are missing or invalid.
    """
    # helper to look up extended header values by tag
    def get_tag(tag):
        for entry in ext_entries:
            if entry["tag"] == tag:
                return entry["value"]
        return None

    # s - scalar
    # c - complex
    # v - vector
    # q - quad - TODO: pri 2 - add support for other types if they are commonly used.
    #
    # b: 8-bit integer
    # i: 16-bit integer
    # l: 32-bit integer
    # x: 64-bit integer
    # f: 32-bit float
    # d: 64-bit float

    # global datatype object - little endian
    datatype_map_le = {
        "SB": "ri8_le",
        "SI": "ri16_le",
        "SL": "ri32_le",
        "SX": "ri64_le",
        "SF": "rf32_le",
        "SD": "rf64_le",
        "CB": "ci8_le",
        "CI": "ci16_le",
        "CL": "ci32_le",
        "CX": "ci64_le",
        "CF": "cf32_le",
        "CD": "cf32_le",
    }

    # global datatype object - big endian
    datatype_map_be = {
        "SB": "ri8_be",
        "SI": "ri16_be",
        "SL": "ri32_be",
        "SX": "ri64_be",
        "SF": "rf32_be",
        "SD": "rf64_be",
        "CB": "ci8_be",
        "CI": "ci16_be",
        "CL": "ci32_be",
        "CX": "ci64_be",
        "CF": "cf32_be",
        "CD": "cf32_be",
    }

    # header data representation: 'EEEI' or 'IEEE' (little or big data endianess representation)
    header_rep = hcb.get("head_rep")

    # data_rep: 'EEEI' or 'IEEE' (little or big data endianess representation)
    data_rep = hcb.get("data_rep")

    # data_format: for example 'CI' or 'SD' (data format code - real or complex, int or float)
    data_format = hcb.get("format")

    if data_rep == "EEEI":  # little endian
        data_map = datatype_map_le.get(data_format)
    elif data_rep == "IEEE":  # big endian
        data_map = datatype_map_be.get(data_format)

    datatype = data_map if data_map is not None else "unknown"

    log.info(f"determined SigMF datatype: {datatype} and data representation: {data_rep}")

    # sample rate: prefer adjunct.xdelta, else extended header SAMPLE_RATE
    if "adjunct" in hcb and "xdelta" in hcb["adjunct"]:
        sample_rate_hz = 1.0 / hcb["adjunct"]["xdelta"]
    else:
        sample_rate_tag = get_tag("SAMPLE_RATE")
        sample_rate_hz = float(sample_rate_tag) if sample_rate_tag is not None else None

    # for now define static values. perhaps take as JSON input
    hardware_description = "Blue File Conversion - Unknown Hardware"
    blue_author = "Blue File Conversion - Unknown Author"
    blue_license = "Blue File Conversion - Unknown License"

    if "outlets" in hcb and hcb["outlets"] > 0:
        num_channels = int(hcb["outlets"])
    else:
        num_channels = 1

    # base global metadata
    global_md = {
        "core:author": blue_author,
        "core:datatype": datatype,
        "core:description": hcb.get("keywords", ""),
        "core:hw": hardware_description,
        "core:license": blue_license,
        "core:num_channels": num_channels,
        "core:sample_rate": sample_rate_hz,
        "core:version": "1.0.0",
    }

    for name, _, _, _, desc in HCB_LAYOUT:
        value = hcb.get(name)  # safe access
        if value is None:
            continue  # or set a default
        global_md[f"core:blue_hcb_{name}"] = value

    # merge adjunct fields
    adjunct = hcb.get("adjunct", {})
    for key, value in adjunct.items():
        global_md[f"core:blue_adjunct_header_{key}"] = value

    # merge extended header fields
    for entry in ext_entries:
        name = entry.get("tag")
        if name is None:
            continue
        key = f"core:blue_extended_header_{name}"
        value = entry.get("value")
        if hasattr(value, "item"):
            value = value.item()
        global_md[key] = value

    # convert the datetime object to an ISO 8601 formatted string
    epoch_time_raw = int(hcb.get("timecode", 0))

    # adjust for Bluefile POSIX epoch (1950 vs 1970)
    bluefile_epoch_offset = 631152000  # seconds between 1950 and 1970
    epoch_time = epoch_time_raw - bluefile_epoch_offset

    dt_object_utc = datetime.fromtimestamp(epoch_time, tz=timezone.utc)
    # format with milliseconds and Zulu suffix
    iso_8601_string = dt_object_utc.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    log.debug(f"epoch time: {epoch_time}")
    log.info(f"ISO 8601 time: {iso_8601_string}")

    # captures array
    captures = [
        {
            "core:datetime": iso_8601_string,
            "core:frequency": float(get_tag("RF_FREQ") or 0.0),
            "core:sample_start": 0,
        }
    ]

    # compute SHA‑512 hash of data file
    def compute_sha512(path, bufsize=1024 * 1024):
        """Compute SHA-512 hash of a file in chunks."""
        hash_obj = hashlib.sha512()
        with open(path, "rb") as handle:
            chunk = handle.read(bufsize)
            while chunk:
                hash_obj.update(chunk)
                chunk = handle.read(bufsize)
        return hash_obj.hexdigest()

    # strip the extension from the original file path
    base_file_name = os.path.splitext(file_path)[0]

    # build the .sigmf-data path
    data_file_path = base_file_name + ".sigmf-data"

    # compute SHA-512 of the data file
    data_sha512 = compute_sha512(data_file_path)  # path to the .sigmf-data file
    global_md["core:sha512"] = data_sha512

    # annotations array
    datatype_sizes_bytes = {
        "ri8_le": 1,
        "ri16_le": 2,
        "ri32_le": 4,
        "ci16_le": 4,
        "ci32_le": 8,
        "cf32_le": 8,
        "rf32_le": 4,
        "rf64_le": 8,
        "ri64_le": 8,
        "ri8_be": 1,
        "ri16_be": 2,
        "ri32_be": 4,
        "ci16_be": 4,
        "ci32_be": 8,
        "cf32_be": 8,
        "rf32_be": 4,
        "rf64_be": 8,
        "ri64_be": 8,
    }

    # calculate sample count
    data_size_bytes = int(hcb.get("data_size", 0))
    if datatype not in datatype_sizes_bytes:
        raise ValueError(f"Unsupported datatype {datatype}")
    bytes_per_sample = datatype_sizes_bytes[datatype]
    sample_count = int(data_size_bytes // bytes_per_sample)

    rf_freq_hz = float(get_tag("RF_FREQ") or 0.0)
    bandwidth_hz = float(get_tag("SBT_BANDWIDTH") or 0.0)

    annotations = [
        {
            "core:sample_start": 0,
            "core:sample_count": sample_count,
            "core:freq_upper_edge": rf_freq_hz + bandwidth_hz,
            "core:freq_lower_edge": rf_freq_hz,
            "core:label": "Sceptere",
        }
    ]

    # final SigMF object
    sigmf_metadata = {
        "global": global_md,
        "captures": captures,
        "annotations": annotations,
    }

    # write .sigmf-meta file
    base_file_name = os.path.splitext(file_path)[0]
    meta_path = base_file_name + ".sigmf-meta"

    with open(meta_path, "w") as meta_handle:
        json.dump(sigmf_metadata, meta_handle, indent=2)
    log.info(f"wrote SigMF metadata to {meta_path}")

    return sigmf_metadata


def blue_file_to_sigmf(file_path):
    """
    Convert a MIDIS Bluefile to SigMF metadata and data.

    Parameters
    ----------
    file_path : str
        Path to the Blue file.

    Returns
    -------
    numpy.ndarray
        IQ Data.
    """
    log.info("starting blue file processing")

    # read Header control block (HCB) from blue file to determine how to process the rest of the file
    hcb = read_hcb(file_path)

    log.debug("Header Control Block (HCB) Fields")
    for name, _, _, _, desc in HCB_LAYOUT:
        log.debug(f"{name:10s}: {hcb[name]!r}  # {desc}")

    log.debug("Adjunct Header")
    log.debug(hcb.get("adjunct", hcb.get("adjunct_raw")))

    # data_rep: 'EEEI' or 'IEEE' (little or big extended header endianness representation)
    extended_header_endianness = hcb.get("head_rep")

    if extended_header_endianness == "EEEI":
        ext_endianness = "<"  # little-endian
    elif extended_header_endianness == "IEEE":
        ext_endianness = ">"  # big-endian
    else:
        raise ValueError(f"Unknown head_rep value: {extended_header_endianness}")

    # parse extended header entries
    ext_entries = parse_extended_header(file_path, hcb, ext_endianness)
    log.debug("Extended Header Keywords")
    for entry in ext_entries:
        log.debug(f"{entry['tag']:20s}:{entry['value']}")
    log.info(f"total extended header entries: {len(ext_entries)}")

    # data_rep: 'EEEI' or 'IEEE' (little or big data endianness representation)
    data_rep_endianness = hcb.get("data_rep")
    data_endianness = "<" if data_rep_endianness == "EEEI" else ">"

    # parse key data values
    # iq_data will be available if needed for further processing.
    try:
        iq_data = parse_data_values(file_path, hcb, data_endianness)
    except Exception as error:
        raise RuntimeError(f"Failed to parse data values: {error}") from error

    # call the SigMF conversion for metadata generation
    blue_to_sigmf(hcb, ext_entries, file_path)

    # return the IQ data if needed for further processing if needed
    return iq_data


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

    blue_file_to_sigmf(args.input)

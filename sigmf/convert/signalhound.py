# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Signal Hound Converter"""

import getpass
import io
import logging
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as ET
import numpy as np

from .. import SigMFFile, fromfile
from ..error import SigMFConversionError
from ..sigmffile import get_sigmf_filenames
from ..utils import SIGMF_DATETIME_ISO8601_FMT

log = logging.getLogger()


def _text_of(root: Element, tag: str) -> Optional[str]:
    """Extract and strip text from XML element."""
    elem = root.find(tag)
    return elem.text.strip() if (elem is not None and elem.text is not None) else None


def _parse_preview_trace(text: Optional[str]) -> List[float]:
    """
    Preview trace is a max-hold trace of the signal power across the capture, represented as a comma-separated string of values.

    Example
    -------
    >>> trace_str = "-1.0, 0.1, 0.5, 0.3, 0.7"
    >>> _parse_preview_trace(trace_str)
    [-1.0, 0.1, 0.5, 0.3, 0.7]
    """
    if text is None:
        return []
    s = text.strip()
    if s.endswith(","):
        s = s[:-1]
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = []
    for part in parts:
        vals.append(float(part))
    return vals


def validate_spike(xml_path: Path) -> None:
    """
    Validate required Spike XML metadata fields and associated IQ file.

    Parameters
    ----------
    xml_path : Path
        Path to the Spike XML file.

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid, or IQ file doesn't exist.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # validate CenterFrequency
    center_freq_raw = _text_of(root, "CenterFrequency")
    try:
        center_frequency = float(center_freq_raw)
    except (TypeError, ValueError) as err:
        raise SigMFConversionError(f"Invalid or missing CenterFrequency: {center_freq_raw}") from err

    # validate SampleRate
    sample_rate_raw = _text_of(root, "SampleRate")
    try:
        sample_rate = float(sample_rate_raw)
    except (TypeError, ValueError) as err:
        raise SigMFConversionError(f"Invalid or missing SampleRate: {sample_rate_raw}") from err

    if sample_rate <= 0:
        raise SigMFConversionError(f"Invalid SampleRate: {sample_rate} (must be > 0)")

    # validate DataType
    data_type_raw = _text_of(root, "DataType")
    if data_type_raw is None:
        raise SigMFConversionError("Missing DataType in Spike XML")

    # check datatype mapping - currently only "Complex Short" is supported
    if data_type_raw != "Complex Short":
        raise SigMFConversionError(f"Unsupported Spike DataType: {data_type_raw}")

    # validate associated IQ file exists
    iq_file_path = xml_path.with_suffix(".iq")
    if not iq_file_path.exists():
        raise SigMFConversionError(f"Could not find associated IQ file: {iq_file_path}")

    # validate IQ file size is aligned to sample boundary
    filesize = iq_file_path.stat().st_size
    elem_size = np.dtype(np.int16).itemsize
    frame_bytes = 2 * elem_size  # I and Q components
    if filesize % frame_bytes != 0:
        raise SigMFConversionError(f"IQ file size {filesize} not divisible by {frame_bytes}; partial sample present")


def _build_metadata(xml_path: Path) -> Tuple[dict, dict, list, int]:
    """
    Build SigMF metadata components from the Spike XML file.

    Parameters
    ----------
    xml_path : Path
        Path to the Spike XML file.

    Returns
    -------
    tuple of (dict, dict, list, int)
        global_info, capture_info, annotations, sample_count

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    log.info("converting spike xml metadata to sigmf format")

    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # validate required fields and associated IQ file
    validate_spike(xml_path)

    # extract and convert required fields
    center_frequency = float(_text_of(root, "CenterFrequency"))
    sample_rate = float(_text_of(root, "SampleRate"))
    data_type_raw = _text_of(root, "DataType")

    # optional EpochNanos field
    epoch_nanos = None
    epoch_nanos_raw = _text_of(root, "EpochNanos")
    if epoch_nanos_raw:
        try:
            epoch_nanos = int(epoch_nanos_raw)
        except ValueError:
            log.warning(f"could not parse EpochNanos: {epoch_nanos_raw}")

    # map datatype
    if data_type_raw == "Complex Short":
        data_type = "ci16_le"  # complex int16 little-endian
    else:
        raise SigMFConversionError(f"Unsupported Spike DataType: {data_type_raw}")

    # optional fields - only convert if present and valid
    reference_level = None
    reference_level_raw = _text_of(root, "ReferenceLevel")
    if reference_level_raw:
        try:
            reference_level = float(reference_level_raw)
        except ValueError:
            log.warning(f"could not parse ReferenceLevel: {reference_level_raw}")

    decimation = None
    decimation_raw = _text_of(root, "Decimation")
    if decimation_raw:
        try:
            decimation = int(float(decimation_raw))
        except ValueError:
            log.warning(f"could not parse Decimation: {decimation_raw}")

    if_bandwidth = None
    if_bandwidth_raw = _text_of(root, "IFBandwidth")
    if if_bandwidth_raw:
        try:
            if_bandwidth = float(if_bandwidth_raw)
        except ValueError:
            log.warning(f"could not parse IFBandwidth: {if_bandwidth_raw}")

    scale_factor = None
    scale_factor_raw = _text_of(root, "ScaleFactor")
    if scale_factor_raw:
        try:
            scale_factor = float(scale_factor_raw)
        except ValueError:
            log.warning(f"could not parse ScaleFactor: {scale_factor_raw}")

    device_type = _text_of(root, "DeviceType")
    serial_number = _text_of(root, "SerialNumber")
    iq_file_name = _text_of(root, "IQFileName")

    # parse preview trace if present
    preview_trace_raw = _text_of(root, "PreviewTrace")
    preview_trace = _parse_preview_trace(preview_trace_raw) if preview_trace_raw else None

    # build hardware description with available information
    hw_parts = []
    if device_type:
        hw_parts.append(f"{device_type}")
    else:
        hw_parts.append("Signal Hound Device")

    if serial_number:
        hw_parts.append(f"S/N: {serial_number}")

    if decimation:
        hw_parts.append(f"decimation: {decimation}")

    hardware_description = ", ".join(hw_parts) if hw_parts else "Signal Hound Device"

    # strip the extension from the original file path
    base_file_name = xml_path.with_suffix("")
    # build the .iq file path for data file
    data_file_path = base_file_name.with_suffix(".iq")
    filesize = data_file_path.stat().st_size

    # complex 16-bit integer IQ data > ci16_le in SigMF
    elem_size = np.dtype(np.int16).itemsize
    frame_bytes = 2 * elem_size  # I and Q components

    # calculate sample count using the original IQ data file size
    sample_count_calculated = filesize // frame_bytes
    log.debug("sample count: %d", sample_count_calculated)

    # convert the datetime object to an ISO 8601 formatted string if EpochNanos is present
    iso_8601_string = None
    if epoch_nanos is not None:
        secs = epoch_nanos // 1_000_000_000
        rem_ns = epoch_nanos % 1_000_000_000
        dt = datetime.fromtimestamp(secs, tz=timezone.utc) + timedelta(microseconds=rem_ns / 1000)
        iso_8601_string = dt.strftime(SIGMF_DATETIME_ISO8601_FMT)

    # base global metadata
    global_md = {
        SigMFFile.AUTHOR_KEY: getpass.getuser(),
        SigMFFile.DATATYPE_KEY: data_type,
        SigMFFile.HW_KEY: hardware_description,
        SigMFFile.NUM_CHANNELS_KEY: 1,
        SigMFFile.RECORDER_KEY: "Official SigMF Signal Hound converter",
        SigMFFile.SAMPLE_RATE_KEY: sample_rate,
        SigMFFile.EXTENSIONS_KEY: [{"name": "spike", "version": "0.0.1", "optional": True}],
    }

    # add optional spike-specific fields to global metadata using spike: namespace
    # only include fields that aren't already represented in standard SigMF metadata
    if reference_level:
        global_md["spike:reference_level_dbm"] = reference_level
    if scale_factor:
        global_md["spike:scale_factor_mw"] = scale_factor  # to convert raw to mW
    if if_bandwidth:
        global_md["spike:if_bandwidth_hz"] = if_bandwidth
    if iq_file_name:
        global_md["spike:iq_filename"] = iq_file_name  # provenance
    if preview_trace:
        global_md["spike:preview_trace"] = preview_trace  # max-hold trace

    # capture info
    capture_info = {
        SigMFFile.FREQUENCY_KEY: center_frequency,
    }
    if iso_8601_string:
        capture_info[SigMFFile.DATETIME_KEY] = iso_8601_string

    # create annotations array using calculated values
    annotations = []
    if if_bandwidth:
        upper_frequency_edge = center_frequency + (if_bandwidth / 2.0)
        lower_frequency_edge = center_frequency - (if_bandwidth / 2.0)
        annotations.append(
            {
                SigMFFile.START_INDEX_KEY: 0,
                SigMFFile.LENGTH_INDEX_KEY: sample_count_calculated,
                SigMFFile.FLO_KEY: lower_frequency_edge,
                SigMFFile.FHI_KEY: upper_frequency_edge,
                SigMFFile.LABEL_KEY: "Spike",
            }
        )

    return global_md, capture_info, annotations, sample_count_calculated


def convert_iq_data(xml_path: Path, sample_count: int) -> np.ndarray:
    """
    Convert IQ data in .iq file to SigMF based on values in Spike XML file.

    Parameters
    ----------
    xml_path : Path
        Path to the spike XML file.
    sample_count : int
        Number of samples to read.

    Returns
    -------
    numpy.ndarray
        Parsed samples.
    """
    log.debug("parsing spike file data values")
    iq_file_path = xml_path.with_suffix(".iq")

    # calculate element count (I and Q samples)
    elem_count = sample_count * 2  # *2 for I and Q samples

    # complex 16-bit integer IQ data > ci16_le in SigMF
    elem_size = np.dtype(np.int16).itemsize

    # read raw interleaved int16 IQ
    samples = np.fromfile(iq_file_path, dtype=np.int16, offset=0, count=elem_count)

    # trim trailing partial bytes
    if samples.nbytes % elem_size != 0:
        trim = samples.nbytes % elem_size
        log.warning("trimming %d trailing byte(s) to align samples", trim)
        samples = samples[: -(trim // elem_size)]

    return samples


def _add_annotations(meta: SigMFFile, annotations: list) -> None:
    for annotation in annotations:
        start_idx = annotation.get(SigMFFile.START_INDEX_KEY, 0)
        length = annotation.get(SigMFFile.LENGTH_INDEX_KEY)
        annot_metadata = {
            k: v for k, v in annotation.items() if k not in [SigMFFile.START_INDEX_KEY, SigMFFile.LENGTH_INDEX_KEY]
        }
        meta.add_annotation(start_idx, length=length, metadata=annot_metadata)


def signalhound_to_sigmf(
    signalhound_path: Path,
    out_path: Optional[Path] = None,
    create_archive: bool = False,
    create_ncd: bool = False,
    overwrite: bool = False,
) -> SigMFFile:
    """
    Read a signalhound file, optionally write sigmf archive, return associated SigMF object.

    Parameters
    ----------
    signalhound_path : Path
        Path to the signalhound file.
    out_path : Path, optional
        Path to the output SigMF metadata file.
    create_archive : bool, optional
        When True, package output as a .sigmf archive.
    create_ncd : bool, optional
        When True, create Non-Conforming Dataset
    overwrite : bool, optional
        If False, raise exception if output files already exist.

    Returns
    -------
    SigMFFile
        SigMF object, potentially as Non-Conforming Dataset.

    Raises
    ------
    SigMFConversionError
        If the signalhound file cannot be read.
    """
    signalhound_path = Path(signalhound_path)
    out_path = None if out_path is None else Path(out_path)

    # auto-enable NCD when no output path is specified
    if out_path is None:
        create_ncd = True

    # call the SigMF conversion for metadata generation
    global_info, capture_info, annotations, sample_count = _build_metadata(signalhound_path)

    # get filenames for metadata, data, and archive based on output path and input file name
    if out_path is None:
        base_path = signalhound_path
    else:
        base_path = Path(out_path)

    filenames = get_sigmf_filenames(base_path)

    # create NCD if specified, otherwise create standard SigMF dataset or archive
    if create_ncd:
        # spike files have no header or trailing bytes
        global_info[SigMFFile.DATASET_KEY] = signalhound_path.with_suffix(".iq").name
        global_info[SigMFFile.TRAILING_BYTES_KEY] = 0
        capture_info[SigMFFile.HEADER_BYTES_KEY] = 0

        # build the .iq file path for data file
        base_file_name = signalhound_path.with_suffix("")
        data_file_path = base_file_name.with_suffix(".iq")

        # create metadata-only SigMF for NCD pointing to original file
        meta = SigMFFile(global_info=global_info)
        meta.set_data_file(data_file=data_file_path, offset=0)
        meta.data_buffer = io.BytesIO()
        meta.add_capture(0, metadata=capture_info)
        _add_annotations(meta, annotations)

        # write metadata file if output path specified
        if out_path is not None:
            output_dir = filenames["meta_fn"].parent
            output_dir.mkdir(parents=True, exist_ok=True)
            meta.tofile(filenames["meta_fn"], overwrite=overwrite)
            log.info("wrote SigMF non-conforming metadata to %s", filenames["meta_fn"])

        log.debug("created %r", meta)
        return meta

    # create archive if specified, otherwise write separate meta and data files
    if create_archive:
        # use temporary directory for data file when creating archive
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / filenames["data_fn"].name

            # convert iq data and write to temp directory
            try:
                iq_data = convert_iq_data(signalhound_path, sample_count)
            except Exception as e:
                raise SigMFConversionError(f"Failed to convert or parse IQ data values: {e}") from e

            # write converted iq data to temporary file
            iq_data.tofile(data_path)
            log.debug("wrote converted iq data to %s", data_path)

            meta = SigMFFile(data_file=data_path, global_info=global_info)
            meta.add_capture(0, metadata=capture_info)
            _add_annotations(meta, annotations)

            output_dir = filenames["archive_fn"].parent
            output_dir.mkdir(parents=True, exist_ok=True)
            meta.tofile(filenames["archive_fn"], overwrite=overwrite)
            log.info("wrote SigMF archive to %s", filenames["archive_fn"])
            # metadata returned should be for this archive
            meta = fromfile(filenames["archive_fn"])

    else:
        # write separate meta and data files
        # convert iq data for spike file
        try:
            iq_data = convert_iq_data(signalhound_path, sample_count)
        except Exception as e:
            raise SigMFConversionError(f"Failed to convert or parse IQ data values: {e}") from e

        # write data file
        output_dir = filenames["data_fn"].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        iq_data.tofile(filenames["data_fn"])
        log.debug("wrote SigMF dataset to %s", filenames["data_fn"])

        # create sigmffile with converted iq data
        meta = SigMFFile(data_file=filenames["data_fn"], global_info=global_info)
        meta.add_capture(0, metadata=capture_info)
        _add_annotations(meta, annotations)

        # write metadata file
        meta.tofile(filenames["meta_fn"], overwrite=overwrite)
        log.info("wrote SigMF metadata to %s", filenames["meta_fn"])

    log.debug("created %r", meta)
    return meta

# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# last updated 3-16-26

"""converter for signalhound files to SigMF format."""

import os
import json
import io
import logging
import tempfile
from pathlib import Path
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta

import xml.etree.ElementTree as ET
from typing import Optional, List

from .. import SigMFFile
from ..error import SigMFConversionError

from .. import __version__ as toolversion
from .. import fromfile
from ..sigmffile import get_sigmf_filenames
from ..utils import SIGMF_DATETIME_ISO8601_FMT

import sys

# Use logging defined in __main__.py
log = logging.getLogger()

def _to_float(x)  -> Optional[float]:
    """Convert value to float, return None if invalid."""
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x) -> Optional[int]:
    """Convert value to int, return None if invalid."""
    try:
        return int(float(x))
    except Exception:
        return None

def _parse_preview_trace(text) -> List[float]:
    """Parse PreviewTrace string into list of floats."""
    if text is None:
        return []
    s = text.strip()
    if s.endswith(","):
        s = s[:-1]
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    # return both list and numpy array if caller wants either
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            # skip malformed entries
            continue
    return vals


def spike_to_sigmf_metadata(xml_file_path) -> dict:
    """
    Build a SigMF metadata file the spike xml file.

    Parameters
    ----------
    xml_file_path : str
        Path to the spike xml file.
    Returns
    -------
    dict
        SigMF metadata structure.
    # TODO:-> SigMFFile:
    sigmfObj : SigMFFile 
         SigMF object with metadata information.    

    Raises
    ------
    SigMFConversionError
        If required fields are missing or invalid.
    """
    log.info("===== Converting Spike XML metadata to SigMF format =====")

    xml_path = Path(xml_file_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def text_of(tag) -> Optional[str]:
        """Extract and strip text from XML element."""
        el = root.find(tag)
        return el.text.strip() if (el is not None and el.text is not None) else None

    # TODO: Determine if data dictionary is best structure for this data 
 
    md = {}
    # Signal Hound data elements
    for tag in (
        "DeviceType",
        "SerialNumber",
        "DataType",
        "ReferenceLevel",
        "CenterFrequency",
        "SampleRate",
        "Decimation",
        "IFBandwidth",
        "ScaleFactor",
        "IQFileName",
        "EpochNanos",
        "SampleCount",
        "PreviewTrace",
    ):
        md[f"{tag}_raw"] = text_of(tag)
    
    # Optional log.info of data for debug >> 
    # log.info(md)

    # Typed fields / normalized
    md["DataType"] = md.pop("DataType_raw")
    md["DeviceType"] = md.pop("DeviceType_raw")
    md["CenterFrequency"] = _to_float(md.pop("CenterFrequency_raw"))
    md["SampleCount"] = _to_int(md.pop("SampleCount_raw"))
    md["SampleRate"] = _to_float(md.pop("SampleRate_raw"))
    md["EpochNanos"] = _to_int(md.pop("EpochNanos_raw"))
    # Will be added as comments or annotations
    md["ReferenceLevel"] = _to_float(md.pop("ReferenceLevel_raw"))
    md["Decimation"] = _to_int(md.pop("Decimation_raw"))
    md["IFBandwidth"] = _to_float(md.pop("IFBandwidth_raw"))
    md["ScaleFactor"] = _to_float(md.pop("ScaleFactor_raw"))
    md["SerialNumber"] = md.pop("SerialNumber_raw")
    md["IQFileName"] = md.pop("IQFileName_raw")

    # PreviewTrace: list of floats and numpy array
    # TODO: Consider adding a flag to include preview trace or not.
    # TODO: Confirm np.int16 data type for preview data elements.
    preview_raw = text_of("PreviewTrace")
    md["PreviewTrace_list"] = _parse_preview_trace(preview_raw)
    md["PreviewTrace_array"] = np.array(md["PreviewTrace_list"], dtype=np.int16)

    # Create a reference to the spike XML data 
    spike_xml = md

    # TODO: Confirm Zero Span Spike files are single channel
    channel_number = 1

    # Check datatype mapping based on Spike XML DataType field - should be "Complex Short"
    spike_data_type = spike_xml.get("DataType")
    if spike_data_type == "Complex Short":
        data_type= "ci16_le"  # complex int16 little-endian
    else:
        raise SigMFConversionError(f"Unsupported Spike DataType: {spike_data_type}")
    # Check for DeviceType field for hardware description, otherwise use generic description
    device_type = spike_xml.get("DeviceType")
    hardware_description = (
        device_type if device_type is not None else "Signal Hound Device"
    )

    # Strip the extension from the original file path
    base_file_name = os.path.splitext(xml_file_path)[0]
    # Build the .iq file path for data file
    data_file_path = base_file_name + ".iq"
    filesize = os.path.getsize(data_file_path)
    # complex 16-bit integer  IQ data > ci16_le in SigMF
    elem_size = np.dtype(np.int16).itemsize
    elem_count = filesize // elem_size
    log.info(f"Element Count: {elem_count}")
    frame_bytes = 2 * elem_size
    if filesize % frame_bytes != 0:
        raise SigMFConversionError(f"File size {filesize} not divisible by {frame_bytes}; partial sample present")

    # Calculate sample count using the original IQ data file size
    sample_count = filesize // frame_bytes
    log.info(f"Sample count: {sample_count}")
    
    # For now define static values. Perhaps take as JSON or command arg input in the future.
    spike_author = "Spike File Conversion - Unknown Author"
    spike_licence = "Spike File Conversion - Unknown License"
    spike_description = "Signal Hound Spike Zero Span File converted to SigMF format"

    # Convert the datetime object to an ISO 8601 formatted string
    epoch_time = spike_xml.get("EpochNanos")
    if epoch_time is None:
        raise SigMFConversionError("Missing EpochNanos in Spike XML")
    epoch_nanos = int(epoch_time)
    secs = epoch_nanos // 1_000_000_000
    rem_ns = epoch_nanos % 1_000_000_000
    dt = datetime.fromtimestamp(secs, tz=timezone.utc) + timedelta(microseconds=rem_ns / 1000)
    iso_8601_string = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    # TODO: Confirm freq_upper_edge and  lower_frequency_edge calculations - is this correct for Spike files? ScaleFactor? Mhz? /2?
    center = float(spike_xml.get("CenterFrequency") or 0.0)
    bandwidth = float(spike_xml.get("IFBandwidth") or 0.0)
    upper_frequency_edge = center + (bandwidth / 2.0)
    lower_frequency_edge = center - (bandwidth / 2.0)

    # --- Base Global Metadata ---
    global_md = {
        "core:author": spike_author,
        "core:datatype": data_type,
        "core:description": spike_description,
        "core:hw": hardware_description,
        "core:license": spike_licence,
        "core:num_channels": channel_number,
        "core:sample_rate": spike_xml.get("SampleRate"),
        "core:version": "1.0.0",
        "core:spike_ReferenceLevel": spike_xml.get("ReferenceLevel"),
        "core:spike_Decimation": spike_xml.get("Decimation"),
        "core:spike_IFBandwidth": spike_xml.get("IFBandwidth"),
        "core:spike_ScaleFactor": spike_xml.get("ScaleFactor"),
        "core:spike_IQFileName": spike_xml.get("IQFileName"),
    }

    # --- Captures array ---
    captures = [
        {
            "core:datetime": iso_8601_string,
            "core:frequency": float(spike_xml.get("CenterFrequency")),
            "core:sample_start": 0,
        }
    ]

    # --- Create annotations array using calculated values---
    annotations = [
        {
            "core:sample_start": 0,
            "core:sample_count": sample_count,
            "core:freq_upper_edge": upper_frequency_edge,
            "core:freq_lower_edge": lower_frequency_edge,
            "core:label": "Spike",
        }
    ]

    # --- Final SigMF object ---
    sigmf = {
        "global": global_md,
        "captures": captures,
        "annotations": annotations,
    }

    # TODO: Return SigMFFile object instead of dict
    
    meta = SigMFFile(global_info=sigmf, skip_checksum=True)
    
    # return sigmf
    return meta

def convert_iq_data(xml_file_path, sigmfObj=None) -> np.ndarray:
    """
    Convert IQ data in .iq file to SigMF based on values in Zero Span XML file.

     Parameters
     ----------
     xml_file_path : str
         Path to the spike zero span XML file.
     sigmfObj : SigMFFile
         SigMF object with metadata information.
 
     Returns
     -------
     numpy.ndarray
         Parsed samples.
    """
   
    # TODO: Although this code may not be needed now, this function can be extended in the future to handle multiple channel recordings?
    # (Samples pending for testing with multi-channel Spike files) 
     
    log.info("===== Parsing spike file data values =====")
    base_file_name = os.path.splitext(xml_file_path)[0]
    iq_file_path = base_file_name + ".iq"

    # TODO: Confirm that the data that is used is correct for the Spike files

    # Gather IQ file information from generated SigMF data file 
    if isinstance(sigmfObj, dict):
        sample_rate = (
            sigmfObj.get("global", {}).get("core:sample_rate")
            or sigmfObj.get("global", {}).get("sample_rate")
            or sigmfObj.get("core:sample_rate")
        )
    

    meta_dict = sigmfObj._metadata

    global_info = meta_dict.get("global", {})
    captures = meta_dict.get("captures", [])
    annotations = meta_dict.get("annotations", [])

    # TODO: Why am I no able to access the sample_rate value from the generated SigMF metadata dict? Is this a problem with how the metadata is being generated in the spike_to_sigmf_metadata function?
    # log.info(f"Sample rate from SigMF metadata dict: {sample_rate}")
    # elem_count=(annotations[0].get("core:sample_count", 0))*2   #*2 for I and Q samples
    
    elem_size = np.dtype(np.int16).itemsize # complex 16-bit integer  IQ data > ci16_le in SigMF
    elem_count = os.path.getsize(iq_file_path) // np.dtype(np.int16).itemsize # complex 16-bit integer  IQ data > ci16_le in SigMF
    
    # Read raw interleaved int16 IQ
    samples = np.fromfile(iq_file_path, dtype=np.int16, offset=0, count=elem_count)

    # Trim trailing partial bytes
    if samples.nbytes % elem_size != 0:
        trim = samples % elem_size
        log.warning("Trimming %d trailing byte(s) to align samples", trim)
        samples -= trim

    # TODO: Confirm that there is no need to reassemble interleaved IQ samples
    # samples = raw_samples[::2] + 1j*raw_samples[1::2] # convert to IQIQIQ...

    # TODO: Use consitent file names in output
    # output_dir = filenames["meta_fn"].parent
    samples.tofile(iq_file_path + ".sigmf-data")
    log.info(f"==== Wrote SigMF data to {iq_file_path + '.sigmf-data'} ====")

    # Return the IQ data if needed for further processing if needed in the future. 
    return samples

def signalhound_to_sigmf(
    signalhound_path: str,
    out_path: Optional[str] = None,
    create_archive: bool = False,
    create_ncd: bool = False,
) -> SigMFFile:
    
   
    """
    Read a signalhound file, optionally write sigmf archive, return associated SigMF object.

    Parameters
    ----------
    signalhound_path : str
        Path to the signalhound file.
    out_path : str, optional
        Path to the output SigMF metadata file.
    create_archive : bool, optional
        When True, package output as a .sigmf archive.
    create_ncd : bool, optional
        When True, create Non-Conforming Dataset

    Returns
    -------
    SigMFFile
        SigMF object, potentially as Non-Conforming Dataset.

    Raises
    ------
    SigMFConversionError
        If the signalhound file cannot be read.
    """
    # FOR TESTING
    # create_ncd=True
 
    signalhound_path = Path(signalhound_path)
    out_path = None if out_path is None else Path(out_path)
    base_file_name = os.path.splitext(signalhound_path)[0]
    meta_path = base_file_name + ".sigmf-meta"

    # auto-enable NCD when no output path is specified
    if out_path is None:
        create_ncd = True

    # TODO: Should time be based on file modification time or the EpochNanos field in the XML metadata? 
    # For now using file modification time
    modify_time = signalhound_path.lstat().st_mtime
    signalhound_datetime = datetime.fromtimestamp(modify_time, tz=timezone.utc)

    capture_info = {
        SigMFFile.DATETIME_KEY: signalhound_datetime.strftime(SIGMF_DATETIME_ISO8601_FMT),
    }

    data_bytes = signalhound_path.stat().st_size
    log.info(f"Data Bytes: {data_bytes}")

    # TODO: Update to use SigMF File object instead of dict
    # Call the SigMF conversion for metadata generation (returns dict)
    SigMFMetaData = spike_to_sigmf_metadata(signalhound_path)

    # Use the generated global metadata dict for SigMFFile construction
    # global_info = SigMFMetaData.get("global", {})
    SigMFMeta_dict = SigMFMetaData._metadata
    global_info = SigMFMeta_dict.get("global", {})

    # Set captures and annotations from the generated metadata dict
    captures = SigMFMeta_dict.get("captures", [])
    # Set annotations
    annotations = SigMFMeta_dict.get("annotations", [])

    # create SigMF metadata 
    meta = SigMFFile(global_info=SigMFMetaData)
    meta.data_file = signalhound_path
    
    # TBD - required?
    header_bytes = 0 # No header bytes for raw IQ files, but could be set to non-zero if needed for other file types or future use cases
    capture_info[SigMFFile.HEADER_BYTES_KEY] = header_bytes
    # TODO: Get actual value global_info[SigMFFile.DATATYPE_KEY] = 
    global_info[SigMFFile.DATATYPE_KEY] = "ci16_le"

    # Get filenames for metadata, data, and archive based on output path and input file name
    filenames = get_sigmf_filenames(out_path)

    # Create NCD if specified, otherwise create standard SigMF dataset or archive
    if create_ncd:
       # Write .sigmf-meta file

       if out_path is not None:
            output_dir = filenames["meta_fn"].parent
            output_dir.mkdir(parents=True, exist_ok=True)
            meta.tofile(filenames["meta_fn"], toarchive=False)
            log.info("For NCD: wrote SigMF non-conforming metadata to %s", filenames["meta_fn"])
            log.debug("created %r", meta)

    if out_path is None:
        base_path = signalhound_path.with_suffix(".sigmf")
    else:
        base_path = Path(out_path)

    # Create Archive if specified, otherwise write separate meta and data files
    if create_archive:
        # use temporary directory for data file when creating archive
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / filenames["data_fn"].name
            
            # Convert IQ data and write to temp directory
            try:
                iq_data = convert_iq_data(str(signalhound_path), meta)
            except Exception as e:
                raise SigMFConversionError(f"Failed to convert or parse IQ data values: {e}")
            
            # Write converted IQ data to temporary file
            iq_data.tofile(data_path)
            # log.info(f"Wrote converted IQ data to {data_path}")
            # TODO: Get actual value global_info[SigMFFile.DATATYPE_KEY] = 
            global_info[SigMFFile.DATATYPE_KEY] = "ci16_le"

            meta = SigMFFile(data_file=data_path, global_info=global_info)
            meta.add_capture(0, metadata=capture_info)
            
            # Add annotations from metadata
            annotations = global_info.get("annotations", [])

            for annotation in annotations:
                start_idx = annotation.get(SigMFFile.START_INDEX_KEY, 0)
                length = annotation.get(SigMFFile.LENGTH_INDEX_KEY)
                annot_metadata = {k: v for k, v in annotation.items() 
                                 if k not in [SigMFFile.START_INDEX_KEY, SigMFFile.LENGTH_INDEX_KEY]}
                meta.add_annotation(start_idx, length=length, metadata=annot_metadata)

            meta.tofile(filenames["archive_fn"], toarchive=True)
            log.info("wrote SigMF archive to %s", filenames["archive_fn"])
            # metadata returned should be for this archive
            meta = fromfile(filenames["archive_fn"])

    else:
        # Write separate meta and data files
        # Convert IQ data for Zero span Spike file
        try:
            iq_data = convert_iq_data(str(signalhound_path), meta)
        except Exception as e:
            raise SigMFConversionError(f"Failed to convert or parse IQ data values: {e}")

        # Create SigMFFile with converted IQ data in a BytesIO buffer
        data_buffer = io.BytesIO(iq_data.tobytes())
        global_info[SigMFFile.DATATYPE_KEY] = "ci16_le"
        # TODO: Get actual value global_info[SigMFFile.DATATYPE_KEY] = 
        header_bytes = 0 # No header bytes for raw IQ files, but could be set to non-zero if needed for other file types or future use cases
        capture_info[SigMFFile.HEADER_BYTES_KEY] = header_bytes
        
        meta = SigMFFile(global_info=global_info)
        
        meta.set_data_file(data_buffer=data_buffer, skip_checksum=True)
        meta.add_capture(0, metadata=capture_info)
   
        # Add annotations from metadata
        annotations = global_info.get("annotations", [])

        for annotation in annotations:
            start_idx = annotation.get(SigMFFile.START_INDEX_KEY, 0)
            length = annotation.get(SigMFFile.LENGTH_INDEX_KEY)
          
            # Pass remaining fields as metadata (excluding standard annotation keys)
            annot_metadata = {k: v for k, v in annotation.items() 
                            if k not in [SigMFFile.START_INDEX_KEY, SigMFFile.LENGTH_INDEX_KEY]}
            meta.add_annotation(start_idx, length=length, metadata=annot_metadata)

        # Write metadata and data files
        output_dir = filenames["meta_fn"].parent
        output_dir.mkdir(parents=True, exist_ok=True)
        meta.tofile(filenames["meta_fn"], toarchive=False)
        log.info("wrote SigMF metadata and data files to %s", filenames["meta_fn"])

    log.debug("Created %r", meta)
    return meta

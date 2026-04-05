# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for Rohde and Schwarz Converter"""

import tarfile
from pathlib import Path
import numpy as np
import pytest
from sigmf import SigMFFile
import defusedxml.ElementTree as ET


from sigmf.convert.rohdeschwarz import (
    SigMFConversionError,
    _build_metadata,
    convert_iq_data,
    extract_iq_tar_to_directory,
    validate_rohdeschwarz,
    xml_to_dict,
    rohdeschwarz_to_sigmf,
)

"""
Create a minimal, valid Rohde & Schwarz IQ.TAR file for testing.
"""
def _write_rohdeschwarz_tar(tmp_path: Path, xml_filename: str = "metadata.xml", iq_filename: str = "File.complex.1ch.float32", xml_content: str = None, iq_values: np.ndarray = None):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    if iq_values is None:
        iq_values = np.arange(8, dtype=np.float32)

    iq_path = source_dir / iq_filename
    iq_values.tofile(iq_path)

    if xml_content is None:
        xml_content = f"""<Root>
  <Clock>2400000000.0</Clock>
  <Samples>2000000.0</Samples>
  <ScalingFactor>1.0</ScalingFactor>
  <DataType>float32</DataType>
  <Format>complex</Format>
  <NumberOfChannels>1</NumberOfChannels>
  <DataFilename>{iq_filename}</DataFilename>
  <EpochNanos>1672531200000000000</EpochNanos>
  <Name>Test Stream</Name>
  <Comment>Unit test capture</Comment>
  <UserData>test-userdata</UserData>
  <PreviewData>
    <Magnitude>
      <float>1.0</float>
      <float>2.0</float>
    </Magnitude>
  </PreviewData>
</Root>"""
    xml_path = source_dir / xml_filename
    xml_path.write_text(xml_content, encoding="utf-8")

    tar_path = tmp_path / "rohdeschwarz.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(xml_path, arcname=xml_path.name)
        tar.add(iq_path, arcname=iq_path.name)

    return tar_path


def test_xml_to_dict_repeated_tags_and_nested_elements():
    xml_text = """
    <PreviewData>
        <Magnitude>
            <float>1.0</float>
            <float>2.0</float>
        </Magnitude>
        <Phase>0.0</Phase>
    </PreviewData>
    """
    root = ET.fromstring(xml_text)
    result = xml_to_dict(root)

    assert result["Magnitude"]["float"] == ["1.0", "2.0"]
    assert result["Phase"] == "0.0"


def test_extract_iq_tar_to_directory_extracts_xml_and_iq_file(tmp_path):
    tar_path = _write_rohdeschwarz_tar(tmp_path)
    extract_dir = tmp_path / "extracted"

    xml_path = extract_iq_tar_to_directory(tar_path, extract_dir)

    assert xml_path.exists()
    assert (extract_dir / "File.complex.1ch.float32").exists()
    assert xml_path.read_text(encoding="utf-8").startswith("<Root>")


def test_extract_iq_tar_to_directory_raises_when_no_xml(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    iq_path = source_dir / "File.complex.1ch.float32"
    np.arange(8, dtype=np.float32).tofile(iq_path)

    tar_path = tmp_path / "empty_xml.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(iq_path, arcname=iq_path.name)

    with pytest.raises(FileNotFoundError, match="No XML metadata file found"):
        extract_iq_tar_to_directory(tar_path, tmp_path / "extracted")


def test_validate_rohdeschwarz_raises_on_missing_datafilename(tmp_path):
    xml_path = tmp_path / "missing_data_filename.xml"
    xml_path.write_text(
        """<Root>
  <Clock>2400000000.0</Clock>
  <Samples>2000000.0</Samples>
  <ScalingFactor>1.0</ScalingFactor>
  <DataType>float32</DataType>
  <Format>complex</Format>
</Root>""",
        encoding="utf-8",
    )

    with pytest.raises(SigMFConversionError, match="Missing DataFilename"):
        validate_rohdeschwarz(xml_path)


def test_validate_rohdeschwarz_raises_on_unsupported_datatype(tmp_path):
    xml_path = tmp_path / "unsupported_dtype.xml"
    xml_path.write_text(
        """<Root>
  <Clock>2400000000.0</Clock>
  <Samples>2000000.0</Samples>
  <ScalingFactor>1.0</ScalingFactor>
  <DataType>float64</DataType>
  <Format>complex</Format>
  <DataFilename>File.complex.1ch.float32</DataFilename>
</Root>""",
        encoding="utf-8",
    )
    (tmp_path / "File.complex.1ch.float32").write_bytes(b"\x00" * 32)

    with pytest.raises(SigMFConversionError, match="float64"):
        validate_rohdeschwarz(xml_path)


def test_build_metadata_and_convert_iq_data(tmp_path):
    tar_path = _write_rohdeschwarz_tar(tmp_path)
    extract_dir = tmp_path / "extracted"
    xml_path = extract_iq_tar_to_directory(tar_path, extract_dir)

    global_info, capture_info, annotations, sample_count = _build_metadata(xml_path)

    assert sample_count == 4
    assert global_info[SigMFFile.DATATYPE_KEY] == "cf32_le"
    assert global_info["rohdeschwarz:scaling_factor"] == 1.0
    assert global_info["rohdeschwarz:iq_datafilename"] == "File.complex.1ch.float32"
    assert "rohdeschwarz:preview_trace" in global_info
    assert capture_info[SigMFFile.FREQUENCY_KEY] == 0.0
    assert SigMFFile.DATETIME_KEY in capture_info
    assert annotations[0][SigMFFile.LENGTH_INDEX_KEY] == 4
    assert annotations[0][SigMFFile.LABEL_KEY] == "rohdeschwarz"

    data_file = xml_path.parent / "File.complex.1ch.float32"
    converted = convert_iq_data(data_file, sample_count)

    assert converted.shape == (8,)
    np.testing.assert_array_equal(converted, np.arange(8, dtype=np.float32))


def _global_info(meta: SigMFFile):
    if hasattr(meta, "global_info"):
        return meta.global_info
    return meta.get_global_info()


def test_rohdeschwarz_to_sigmf_create_ncd_returns_metadata_object(tmp_path):
    tar_path = _write_rohdeschwarz_tar(tmp_path)
    meta = rohdeschwarz_to_sigmf(tar_path)
    global_info = _global_info(meta)

    assert global_info[SigMFFile.DATASET_KEY] == "File.complex.1ch.float32"
    assert global_info[SigMFFile.TRAILING_BYTES_KEY] == 0
    assert global_info[SigMFFile.DATATYPE_KEY] == "cf32_le"
    assert global_info["rohdeschwarz:iq_datafilename"] == "File.complex.1ch.float32"

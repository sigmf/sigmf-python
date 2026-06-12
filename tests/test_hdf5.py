# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for the optional hdf5-meta metadata sidecar."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sigmf
from sigmf.error import SigMFFileExistsError

# the entire module depends on the optional h5py dependency
h5py = pytest.importorskip("h5py")

from sigmf import hdf5  # noqa: E402 - imported after the h5py skip guard


def _make_recording(tmp_path):
    """Build a SigMFFile with heterogeneous captures and annotations."""
    data = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
    meta = sigmf.fromarray(data)
    meta.sample_rate = 1e6
    meta.author = "tester@example.com"
    meta.add_capture(0, metadata={sigmf.FREQUENCY_KEY: 915e6})
    # annotation 0: has a float edge field
    meta.add_annotation(0, length=100, metadata={sigmf.LABEL_KEY: "burst_0", sigmf.FREQ_LOWER_EDGE_KEY: 914e6})
    # annotation 1: carries a nested object field (signal:detail)
    meta.add_annotation(
        1000, length=100, metadata={sigmf.LABEL_KEY: "burst_1", "signal:detail": {"type": "digital", "order": 4}}
    )
    # annotation 2: omits sample_count entirely and has a unique field
    meta.add_annotation(2000, metadata={sigmf.COMMENT_KEY: "tail"})
    return meta


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def written(tmp_dir):
    """Write a recording with sidecar; return (tmp_dir, base, json_doc)."""
    meta = _make_recording(tmp_dir)
    base = tmp_dir / "rec"
    meta.tofile(base, write_hdf5=True, skip_validate=True)
    doc = json.loads((tmp_dir / "rec.sigmf-meta").read_text())
    return tmp_dir, base, doc


# ---------------------------------------------------------------------------
# writing + declaration
# ---------------------------------------------------------------------------
def test_sidecar_written_and_declared(written):
    """tofile(write_hdf5=True) writes the sidecar and declares the extension."""
    tmp_dir, _base, doc = written
    assert (tmp_dir / "rec.sigmf-meta").is_file()
    assert (tmp_dir / "rec.sigmf-meta.h5").is_file()

    assert doc["global"]["hdf5-meta:file"] == "rec.sigmf-meta.h5"
    ext = next(e for e in doc["global"]["core:extensions"] if e["name"] == "hdf5-meta")
    assert ext["optional"] is True


def test_default_fromfile_ignores_sidecar(written):
    """Option A: the canonical sigmf.fromfile reads pure JSON, never the sidecar."""
    _tmp_dir, base, doc = written
    loaded = sigmf.fromfile(base)
    assert isinstance(loaded, sigmf.SigMFFile)
    assert loaded.get_annotations() == doc["annotations"]
    assert loaded.get_captures() == doc["captures"]


# ---------------------------------------------------------------------------
# direct module round-trip
# ---------------------------------------------------------------------------
def test_module_roundtrip_equivalence(written):
    """read_hdf5_sidecar reproduces the JSON metadata exactly."""
    tmp_dir, _base, doc = written
    restored = hdf5.read_hdf5_sidecar(tmp_dir / "rec.sigmf-meta.h5")
    assert restored["captures"] == doc["captures"]
    assert restored["annotations"] == doc["annotations"]
    # global gains the hdf5-meta declaration fields, which are real metadata
    assert restored["global"]["core:sample_rate"] == 1e6


# ---------------------------------------------------------------------------
# hdf5.open — zero JSON, columnar fast path
# ---------------------------------------------------------------------------
def test_open_zero_json_columnar(written):
    """hdf5.open reads only the .h5 and serves columns without dict building."""
    tmp_dir, _base, doc = written
    with hdf5.open(tmp_dir / "rec.sigmf-meta.h5") as sf:
        assert isinstance(sf, hdf5.SigMFFileHDF5)
        assert sf.sample_rate == 1e6
        assert sf.num_annotations() == 3
        assert sf.num_captures() == 1

        # absent values come back as None
        assert sf.annotations_column("core:label") == ["burst_0", "burst_1", None]
        assert sf.annotations_column("core:sample_count") == [100, 100, None]
        assert sf.annotations_column("core:sample_start") == [0, 1000, 2000]
        # nested object column decodes back to a dict
        assert sf.annotations_column("signal:detail") == [None, {"type": "digital", "order": 4}, None]


def test_open_structured_array(written):
    """annotations_array returns a numpy structured array with colon-keyed names."""
    tmp_dir, _base, _doc = written
    with hdf5.open(tmp_dir / "rec.sigmf-meta.h5") as sf:
        arr = sf.annotations_array(["core:sample_start", "core:label"])
        assert arr.dtype.names == ("core:sample_start", "core:label")
        assert arr["core:sample_start"].tolist() == [0, 1000, 2000]


def test_open_compat_accessors(written):
    """get_annotations/get_captures/to_sigmffile match the JSON content."""
    tmp_dir, _base, doc = written
    with hdf5.open(tmp_dir / "rec.sigmf-meta.h5") as sf:
        assert sf.get_annotations() == doc["annotations"]
        assert sf.get_captures() == doc["captures"]
        # index-filtered access mirrors SigMFFile.get_annotations(index=...)
        labels = [a["core:label"] for a in sf.get_annotations(index=1050)]
        assert labels == ["burst_1"]
        sff = sf.to_sigmffile()
        assert isinstance(sff, sigmf.SigMFFile)
        assert sff.get_annotations() == doc["annotations"]


def test_unknown_column_raises(written):
    """Requesting a non-existent column raises KeyError."""
    tmp_dir, _base, _doc = written
    with hdf5.open(tmp_dir / "rec.sigmf-meta.h5") as sf:
        with pytest.raises(KeyError):
            sf.annotations_column("core:does_not_exist")


# ---------------------------------------------------------------------------
# hdf5.fromfile — discovery
# ---------------------------------------------------------------------------
def test_fromfile_prefers_sidecar(written):
    """hdf5.fromfile returns the fast reader when a valid sidecar exists."""
    _tmp_dir, base, doc = written
    sf = hdf5.fromfile(base)
    try:
        assert isinstance(sf, hdf5.SigMFFileHDF5)
        assert sf.get_annotations() == doc["annotations"]
    finally:
        sf.close()


def test_fromfile_falls_back_without_sidecar(tmp_dir):
    """hdf5.fromfile returns a plain SigMFFile when no sidecar is declared."""
    data = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    meta = sigmf.fromarray(data)
    meta.add_capture(0)
    base = tmp_dir / "plain"
    meta.tofile(base, skip_validate=True)  # no write_hdf5

    sf = hdf5.fromfile(base)
    assert isinstance(sf, sigmf.SigMFFile)


def test_fromfile_require_sidecar_raises(tmp_dir):
    """require_sidecar=True raises instead of falling back to JSON."""
    data = (np.random.randn(64) + 1j * np.random.randn(64)).astype(np.complex64)
    meta = sigmf.fromarray(data)
    meta.add_capture(0)
    base = tmp_dir / "plain"
    meta.tofile(base, skip_validate=True)

    with pytest.raises(hdf5.SigMFHDF5Error):
        hdf5.fromfile(base, require_sidecar=True)


def test_fromfile_stale_sidecar_warns_and_falls_back(written):
    """A digest mismatch makes hdf5.fromfile warn and use the JSON instead."""
    tmp_dir, base, doc = written

    # edit the JSON without regenerating the sidecar -> stale (relabel in place
    # rather than appending past the data length, to avoid an unrelated warning)
    meta_fn = tmp_dir / "rec.sigmf-meta"
    edited = json.loads(meta_fn.read_text())
    edited["annotations"][0]["core:label"] = "edited_in_json"
    meta_fn.write_text(json.dumps(edited))

    with pytest.warns(UserWarning, match="stale"):
        sf = hdf5.fromfile(base)
    assert isinstance(sf, sigmf.SigMFFile)
    assert sf.get_annotations()[0]["core:label"] == "edited_in_json"


def test_fromfile_corrupt_sidecar_falls_back(written):
    """An unreadable sidecar falls back to JSON (no crash)."""
    tmp_dir, base, doc = written
    (tmp_dir / "rec.sigmf-meta.h5").write_bytes(b"not an hdf5 file")

    sf = hdf5.fromfile(base)
    assert isinstance(sf, sigmf.SigMFFile)
    assert sf.get_annotations() == doc["annotations"]


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_empty_annotations_and_captures(tmp_dir):
    """Recordings with no annotations/captures omit those datasets cleanly."""
    metadata = {
        "global": {sigmf.DATATYPE_KEY: "cf32_le", sigmf.VERSION_KEY: "1.2.0"},
        "captures": [],
        "annotations": [],
    }
    sidecar = tmp_dir / "empty.h5"
    hdf5.write_hdf5_sidecar(metadata, sidecar)

    with h5py.File(sidecar, "r") as handle:
        assert "captures" not in handle
        assert "annotations" not in handle

    with hdf5.open(sidecar) as sf:
        assert sf.num_annotations() == 0
        assert sf.num_captures() == 0
        assert sf.annotations_column("core:sample_start") == []
        assert sf.get_annotations() == []


def test_boolean_and_extensions_roundtrip(tmp_dir):
    """Booleans, arrays, and nested objects in global survive the round trip."""
    metadata = {
        "global": {
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.VERSION_KEY: "1.2.0",
            sigmf.METADATA_ONLY_KEY: True,
            sigmf.EXTENSIONS_KEY: [{"name": "hdf5-meta", "version": "1.0.0", "optional": True}],
        },
        "captures": [],
        "annotations": [],
    }
    sidecar = tmp_dir / "bools.h5"
    hdf5.write_hdf5_sidecar(metadata, sidecar)
    restored = hdf5.read_hdf5_sidecar(sidecar)

    assert restored["global"][sigmf.METADATA_ONLY_KEY] is True
    assert restored["global"][sigmf.EXTENSIONS_KEY] == metadata["global"][sigmf.EXTENSIONS_KEY]


def test_sidecar_overwrite_guard(written):
    """Writing over an existing sidecar without overwrite=True raises."""
    tmp_dir, base, _doc = written
    meta = _make_recording(tmp_dir / "ignored")  # fresh object, same target base
    with pytest.raises(SigMFFileExistsError):
        meta.tofile(base, write_hdf5=True, skip_validate=True, overwrite=False)

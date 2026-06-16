# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
Optional HDF5 metadata sidecar support for the ``hdf5-meta`` SigMF extension.

The ``hdf5-meta`` extension defines an OPTIONAL HDF5 file that stores a
columnar, performance-optimized duplicate of a Recording's metadata. The JSON
``.sigmf-meta`` file remains the complete, authoritative source of truth; the
sidecar is a derived cache that enables faster loads for Recordings with very
large ``captures`` or ``annotations`` arrays.

This module is only imported when HDF5 functionality is requested. It requires
the optional ``h5py`` dependency, installable via ``pip install sigmf[hdf5]``.

See ``extensions/hdf5-meta.sigmf-ext.md`` in the SigMF specification repository
for the on-disk format.
"""

import builtins
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from . import keys
from .error import SigMFError

# extension identity
HDF5_META_EXTENSION = "hdf5-meta"
HDF5_META_VERSION = "1.0.0"

# global fields contributed by this extension (colon notation, as in JSON)
HDF5_META_FILE_KEY = "hdf5-meta:file"
HDF5_META_VERSION_KEY = "hdf5-meta:version"

# default suffix appended to a `.sigmf-meta` filename to form the sidecar name
HDF5_SIDECAR_SUFFIX = ".h5"

# bookkeeping attribute names. These start with "__" and contain no ".", so they
# cannot collide with SigMF "namespace.field" attribute names.
_JSON_ATTRS_HINT = "__json_attrs__"
_JSON_COLUMNS_HINT = "__json_columns__"

# root attribute holding a digest of the authoritative JSON metadata, used to
# detect a stale sidecar (JSON edited without regenerating the .h5).
_SOURCE_DIGEST_ATTR = "source_meta_sha512"


class SigMFHDF5Error(SigMFError):
    """Raised when reading or writing an HDF5 metadata sidecar fails."""


def _require_h5py():
    """Import h5py lazily, raising a helpful error if it is not installed."""
    try:
        import h5py
    except ImportError as exc:
        raise SigMFHDF5Error(
            "HDF5 metadata sidecar support requires the optional 'h5py' "
            "dependency. Install it with: pip install sigmf[hdf5]"
        ) from exc
    return h5py


def _field_to_dot(name):
    """Convert a SigMF field name from colon notation to dot notation.

    Only the first colon (the namespace separator) is converted, so
    ``core:sample_start`` becomes ``core.sample_start``. Field names without a
    colon are returned unchanged.
    """
    return name.replace(":", ".", 1)


def _field_to_colon(name):
    """Inverse of :func:`_field_to_dot`: restore the namespace colon."""
    return name.replace(".", ":", 1)


def _is_scalar_number(value):
    """True for ints/floats but not bool (bool is a subclass of int)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _metadata_digest(metadata):
    """Return a stable SHA-512 hex digest of a SigMF metadata dictionary.

    Keys are sorted so the digest is independent of dict ordering, giving a
    canonical fingerprint of the authoritative JSON content. Used to detect a
    stale sidecar.
    """
    canonical = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha512(canonical).hexdigest()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_hdf5_sidecar(metadata, file_path, compression="gzip"):
    """
    Write a SigMF metadata dictionary to an HDF5 sidecar file.

    Parameters
    ----------
    metadata : dict
        A SigMF metadata dictionary containing ``global``, ``captures``, and
        ``annotations`` keys (as held by ``SigMFFile._metadata``).
    file_path : str | PathLike
        Destination path for the sidecar file.
    compression : str | None, default "gzip"
        Compression filter passed to ``h5py.create_dataset``. Use ``None`` to
        disable compression.
    """
    h5py = _require_h5py()

    global_obj = metadata.get("global", {}) or {}
    captures = metadata.get("captures", []) or []
    annotations = metadata.get("annotations", []) or []

    with h5py.File(file_path, "w") as handle:
        # root attributes
        handle.attrs["sigmf_version"] = str(global_obj.get(keys.VERSION_KEY, ""))
        handle.attrs["hdf5_meta_version"] = HDF5_META_VERSION
        # fingerprint of the authoritative JSON, for stale-sidecar detection
        handle.attrs[_SOURCE_DIGEST_ATTR] = _metadata_digest(metadata)

        # global object -> attributes on /global
        grp = handle.create_group("global")
        _write_global_attrs(grp, global_obj)

        # captures / annotations -> columnar datasets
        if captures:
            _write_records(handle, "captures", captures, compression)
        if annotations:
            _write_records(handle, "annotations", annotations, compression)


def _declare_extension(global_obj, sidecar_filename):
    """Stamp the ``hdf5-meta`` extension fields into a ``global`` object in place.

    Adds an entry to ``core:extensions`` (marked optional, idempotently) and
    sets ``hdf5-meta:file`` / ``hdf5-meta:version``. This mirrors
    ``SigMFFile._declare_hdf5_meta`` so a sidecar generated from raw JSON is
    declared identically to one written via ``SigMFFile.tofile(write_hdf5=True)``.

    Parameters
    ----------
    global_obj : dict
        The SigMF ``global`` object (mutated in place).
    sidecar_filename : str
        Bare filename (not a path) of the ``.h5`` sidecar.
    """
    extensions = global_obj.get(keys.EXTENSIONS_KEY, []) or []
    if not any(ext.get("name") == HDF5_META_EXTENSION for ext in extensions):
        extensions = extensions + [{"name": HDF5_META_EXTENSION, "version": HDF5_META_VERSION, "optional": True}]
        global_obj[keys.EXTENSIONS_KEY] = extensions
    global_obj[HDF5_META_FILE_KEY] = sidecar_filename
    global_obj[HDF5_META_VERSION_KEY] = HDF5_META_VERSION


def generate_sidecar(meta_path, sidecar_path=None, compression="gzip", update_json=True, overwrite=True):
    """
    Generate an HDF5 metadata sidecar from an existing ``.sigmf-meta`` JSON file.

    This is the forward complement of :func:`fromfile`: it reads an
    authoritative JSON Metadata file, writes the columnar ``.h5`` sidecar
    alongside it, and (by default) declares the ``hdf5-meta`` extension in the
    JSON so :func:`fromfile` can discover and digest-verify the sidecar.

    Parameters
    ----------
    meta_path : str | PathLike
        Path to the ``.sigmf-meta`` file (with or without extension). The JSON
        is read once and remains the authoritative source of truth.
    sidecar_path : str | PathLike, optional
        Destination for the sidecar. Defaults to the meta filename with
        ``.h5`` appended (e.g. ``rec.sigmf-meta.h5``), matching the name
        ``SigMFFile.tofile(write_hdf5=True)`` produces.
    compression : str | None, default "gzip"
        Compression filter for the columnar datasets. ``None`` disables it.
    update_json : bool, default True
        If True, stamp ``hdf5-meta:file`` / ``hdf5-meta:version`` and the
        ``core:extensions`` entry into the JSON ``global`` object and rewrite
        the ``.sigmf-meta`` file. When False the JSON is left untouched and the
        sidecar will not be auto-discovered by :func:`fromfile`.
    overwrite : bool, default True
        If False, raise :class:`SigMFHDF5Error` when the sidecar already exists.

    Returns
    -------
    pathlib.Path
        The path to the written sidecar file.

    Raises
    ------
    SigMFHDF5Error
        If the metadata file is missing, unreadable as JSON, or the sidecar
        exists and ``overwrite`` is False.
    """
    from .sigmffile import get_sigmf_filenames

    meta_fn = get_sigmf_filenames(meta_path)["meta_fn"]
    if not meta_fn.is_file():
        raise SigMFHDF5Error(f"Metadata file not found: '{meta_fn}'")

    try:
        with builtins.open(meta_fn, "rb") as fp:
            metadata = json.loads(fp.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise SigMFHDF5Error(f"Could not read SigMF metadata from '{meta_fn}': {exc}") from exc

    if sidecar_path is None:
        sidecar_path = meta_fn.parent / (meta_fn.name + HDF5_SIDECAR_SUFFIX)
    sidecar_path = Path(sidecar_path)

    if sidecar_path.exists() and not overwrite:
        raise SigMFHDF5Error(f"HDF5 sidecar already exists: '{sidecar_path}'")

    if update_json:
        global_obj = metadata.setdefault("global", {})
        _declare_extension(global_obj, sidecar_path.name)
        with builtins.open(meta_fn, "w") as fp:
            json.dump(metadata, fp)

    # write the sidecar from the (now-declared) metadata so its stored digest
    # matches the JSON that fromfile() will verify against
    write_hdf5_sidecar(metadata, sidecar_path, compression=compression)
    return sidecar_path


def _write_global_attrs(grp, global_obj):
    """Store each global key/value pair as an attribute on the /global group.

    Scalars are stored as native attribute types; arrays and objects are
    JSON-encoded strings. The names of JSON-encoded attributes are recorded in
    the ``__json_attrs__`` hint so the reader can decode them unambiguously.
    """
    json_attrs = []
    for key, value in global_obj.items():
        if value is None:
            continue  # null -> omit (datatype mapping)
        attr_name = _field_to_dot(key)
        if isinstance(value, (list, dict)):
            grp.attrs[attr_name] = json.dumps(value)
            json_attrs.append(attr_name)
        else:
            grp.attrs[attr_name] = value
    grp.attrs[_JSON_ATTRS_HINT] = json.dumps(json_attrs)


def _column_dtype(values, present):
    """Decide the storage encoding for one column.

    Returns a tuple ``(numpy_dtype, is_json)``. A column is stored in a native
    numpy dtype only when every row is present and the values are homogeneous
    scalars; otherwise it is promoted to a JSON-encoded string column to
    guarantee an exact round-trip.
    """
    all_present = all(present)
    non_null = [v for v, p in zip(values, present) if p]

    if all_present and non_null:
        if all(isinstance(v, bool) for v in non_null):
            return np.dtype("i1"), False
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in non_null):
            return np.dtype("<i8"), False
        if all(_is_scalar_number(v) for v in non_null) and any(isinstance(v, (float, np.floating)) for v in non_null):
            return np.dtype("<f8"), False
        if all(isinstance(v, str) for v in non_null):
            import h5py

            return h5py.string_dtype(encoding="utf-8"), False

    # mixed presence, mixed/complex types, or empty -> JSON-encoded strings
    import h5py

    return h5py.string_dtype(encoding="utf-8"), True


def _write_records(handle, group_name, records, compression):
    """Write a list-of-dicts SigMF array as a columnar compound HDF5 dataset."""
    # union of all field names across all records, preserving first-seen order
    columns = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                columns.append(key)

    n_rows = len(records)
    col_specs = []  # (json_key, dot_name, numpy_dtype, is_json)
    json_columns = []
    for key in columns:
        values = [record.get(key) for record in records]
        present = [key in record and record[key] is not None for record in records]
        dtype, is_json = _column_dtype(values, present)
        dot_name = _field_to_dot(key)
        col_specs.append((key, dot_name, dtype, is_json))
        if is_json:
            json_columns.append(dot_name)

    compound_dtype = np.dtype([(dot_name, dtype) for _key, dot_name, dtype, _is_json in col_specs])
    array = np.zeros(n_rows, dtype=compound_dtype)

    for row_idx, record in enumerate(records):
        for json_key, dot_name, dtype, is_json in col_specs:
            present = json_key in record and record[json_key] is not None
            if is_json:
                array[dot_name][row_idx] = json.dumps(record[json_key]) if present else ""
            elif not present:
                # sentinel for an absent native value
                if dtype.kind == "f":
                    array[dot_name][row_idx] = np.nan
                # int/string sentinels (0 / "") are only reached when the column
                # is fully present, so this branch is effectively float-only.
            else:
                value = record[json_key]
                if dtype.kind == "b" or dtype == np.dtype("i1"):
                    array[dot_name][row_idx] = 1 if value else 0
                else:
                    array[dot_name][row_idx] = value

    dataset = handle.create_dataset(group_name, data=array, compression=compression)
    dataset.attrs[_JSON_COLUMNS_HINT] = json.dumps(json_columns)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def open_hdf5(file_path):
    """Open an HDF5 sidecar read-only and return the ``h5py.File`` handle.

    The caller is responsible for closing the handle (directly, or via the
    :class:`SigMFFileHDF5` lifecycle).
    """
    h5py = _require_h5py()
    return h5py.File(file_path, "r")


def read_source_digest(handle):
    """Return the stored authoritative-JSON digest, or ``None`` if absent."""
    value = handle.attrs.get(_SOURCE_DIGEST_ATTR)
    return _decode_scalar(value) if value is not None else None


def read_hdf5_sidecar(file_path):
    """
    Read an HDF5 metadata sidecar into a SigMF metadata dictionary.

    Parameters
    ----------
    file_path : str | PathLike
        Path to the ``.h5`` sidecar file.

    Returns
    -------
    dict
        A SigMF metadata dictionary with ``global``, ``captures``, and
        ``annotations`` keys, equivalent to the JSON Metadata file.
    """
    with open_hdf5(file_path) as handle:
        metadata = {"global": {}, "captures": [], "annotations": []}
        if "global" in handle:
            metadata["global"] = read_global_object(handle)
        if "captures" in handle:
            metadata["captures"] = records_from_dataset(handle["captures"])
        if "annotations" in handle:
            metadata["annotations"] = records_from_dataset(handle["annotations"])
    return metadata


def _decode_scalar(value):
    """Convert a numpy/bytes scalar from HDF5 into a native Python type."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_global_object(handle):
    """Reconstruct the SigMF ``global`` object from the ``/global`` group.

    ``handle`` is an open ``h5py.File``. Globals are small and always read
    eagerly.
    """
    if "global" not in handle:
        return {}
    grp = handle["global"]
    json_attrs = set(json.loads(grp.attrs.get(_JSON_ATTRS_HINT, "[]")))
    result = {}
    for attr_name, attr_value in grp.attrs.items():
        if attr_name == _JSON_ATTRS_HINT:
            continue
        key = _field_to_colon(attr_name)
        if attr_name in json_attrs:
            result[key] = json.loads(_decode_scalar(attr_value))
        else:
            result[key] = _decode_scalar(attr_value)
    return result


def record_column_names(dataset):
    """Return the JSON-keyed (colon-notation) field names of a record dataset."""
    return [_field_to_colon(dot_name) for dot_name in dataset.dtype.names]


def read_record_column(dataset, field):
    """
    Read one column of a captures/annotations dataset as a list of values.

    This is the columnar fast path: a single HDF5 column is read and decoded
    without materializing any per-row dictionaries. Absent values (the NaN or
    empty-string sentinel) are returned as ``None`` so the caller can tell
    "field not present in this row" from a real value.

    Parameters
    ----------
    dataset : h5py.Dataset
        A captures or annotations compound dataset.
    field : str
        Field name in JSON colon notation (e.g. ``"core:sample_start"``) or the
        stored dot notation (e.g. ``"core.sample_start"``).

    Returns
    -------
    list
        One entry per row; ``None`` where the field is absent.

    Raises
    ------
    KeyError
        If the field is not a column in the dataset.
    """
    dot_name = _field_to_dot(field)
    if dot_name not in dataset.dtype.names:
        raise KeyError(f"'{field}' is not a column in this dataset")
    json_columns = set(json.loads(dataset.attrs.get(_JSON_COLUMNS_HINT, "[]")))
    column = dataset[dot_name]
    return _decode_column(column, dot_name in json_columns)


def _decode_column(column, is_json):
    """Decode a single numpy column into a list, mapping sentinels to ``None``.

    Shared by the list-of-dicts path (:func:`records_from_dataset`) and the
    columnar path (:func:`read_record_column`) so both apply identical
    NaN/empty-string sentinel handling.
    """
    kind = column.dtype.kind
    if is_json:
        return [(None if text == "" else json.loads(text)) for text in _decode_string_column(column)]
    if kind == "f":
        # NaN marks an absent value (v != v is True only for NaN)
        return [None if v != v else v for v in column.tolist()]
    if kind in ("O", "S", "U"):
        return _decode_string_column(column)
    # integers / booleans-as-int: fully present by construction
    return column.tolist()


def records_from_dataset(dataset):
    """Reconstruct a list-of-dicts SigMF array from a columnar dataset.

    Works column-by-column rather than cell-by-cell: ``ndarray.tolist()`` bulk
    converts a whole column to native Python objects in C, which is far faster
    than indexing numpy scalars one row at a time. This is the compatibility
    path; the fast path (:func:`read_record_column`) avoids building dicts.
    """
    json_columns = set(json.loads(dataset.attrs.get(_JSON_COLUMNS_HINT, "[]")))
    data = dataset[:]
    column_names = data.dtype.names
    n_rows = len(data)

    decoded = [
        (_field_to_colon(dot_name), _decode_column(data[dot_name], dot_name in json_columns))
        for dot_name in column_names
    ]

    records = [{} for _ in range(n_rows)]
    for key, values in decoded:
        for row_idx in range(n_rows):
            value = values[row_idx]
            if value is not None:
                records[row_idx][key] = value
    return records


def _decode_string_column(column):
    """Return a list of Python ``str`` from an HDF5 string column."""
    return [v.decode("utf-8") if isinstance(v, bytes) else v for v in column.tolist()]


def _structured_array(dataset, fields=None):
    """Return a structured ``ndarray`` view of a record dataset.

    Field names are renamed from stored dot notation to SigMF colon notation.
    JSON-encoded columns (nested objects/arrays) are returned as their raw
    encoded strings; callers needing decoded objects should use the list-of-
    dicts path. ``fields`` optionally restricts to a subset (JSON colon names).
    """
    data = dataset[:]
    data.dtype.names = tuple(_field_to_colon(name) for name in data.dtype.names)
    if fields is not None:
        data = data[list(fields)]
    return data


# ---------------------------------------------------------------------------
# fast lazy reader
# ---------------------------------------------------------------------------
class SigMFFileHDF5:
    """
    Lazy, columnar reader for an HDF5 metadata sidecar (the ``hdf5-meta`` fast
    path).

    Unlike :class:`sigmf.sigmffile.SigMFFile`, which stores metadata as
    list-of-dicts, this reader keeps the sidecar open and serves
    captures/annotations as numpy columns/arrays without ever building per-row
    dictionaries. This is the path that is actually faster than parsing JSON.

    The instance holds an open ``h5py.File``; use it as a context manager or
    call :meth:`close` when done::

        with sigmf.hdf5.open("rec.sigmf-meta.h5") as sf:
            starts = sf.annotations_column("core:sample_start")  # ndarray
            labels = sf.annotations_column("core:label")         # ndarray

    For interoperability with code written against ``SigMFFile``, the
    convenience methods :meth:`get_annotations`, :meth:`get_captures`, and
    :meth:`to_sigmffile` materialize list-of-dicts on demand (at JSON speed).
    """

    def __init__(self, handle, global_obj=None, data_file=None):
        """
        Parameters
        ----------
        handle : h5py.File
            An open, readable HDF5 sidecar handle. Ownership transfers to this
            object, which will close it on :meth:`close`.
        global_obj : dict, optional
            The SigMF ``global`` object. If omitted it is read from the
            sidecar's ``/global`` group.
        data_file : str | PathLike, optional
            Path to the associated ``.sigmf-data`` dataset, if known.
        """
        self._handle = handle
        self._global = global_obj if global_obj is not None else read_global_object(handle)
        self.data_file = data_file

    # -- global access (eager; globals are small) --------------------------
    def get_global_info(self):
        """Return the full ``global`` object as a dict."""
        return self._global

    def global_field(self, key, default=None):
        """Return one global field, e.g. ``global_field('core:sample_rate')``."""
        return self._global.get(key, default)

    @property
    def sample_rate(self):
        return self._global.get(keys.SAMPLE_RATE_KEY)

    @property
    def datatype(self):
        return self._global.get(keys.DATATYPE_KEY)

    # -- columnar fast path -------------------------------------------------
    def _dataset(self, name):
        if name not in self._handle:
            return None
        return self._handle[name]

    def num_captures(self):
        """Number of capture segments without materializing them."""
        ds = self._dataset("captures")
        return 0 if ds is None else len(ds)

    def num_annotations(self):
        """Number of annotations without materializing them."""
        ds = self._dataset("annotations")
        return 0 if ds is None else len(ds)

    def capture_field_names(self):
        """JSON-keyed (colon-notation) capture column names."""
        ds = self._dataset("captures")
        return [] if ds is None else record_column_names(ds)

    def annotation_field_names(self):
        """JSON-keyed (colon-notation) annotation column names."""
        ds = self._dataset("annotations")
        return [] if ds is None else record_column_names(ds)

    def captures_column(self, field):
        """Return one capture field across all segments as a list.

        Absent values are ``None``. Raises ``KeyError`` if the field is not a
        column and there are captures, and returns ``[]`` if there are none.
        """
        ds = self._dataset("captures")
        return [] if ds is None else read_record_column(ds, field)

    def annotations_column(self, field):
        """Return one annotation field across all annotations as a list.

        Absent values are ``None``. Raises ``KeyError`` if the field is not a
        column and there are annotations, and returns ``[]`` if there are none.
        """
        ds = self._dataset("annotations")
        return [] if ds is None else read_record_column(ds, field)

    def captures_array(self, fields=None):
        """Return captures as a numpy structured array (colon-keyed columns)."""
        ds = self._dataset("captures")
        if ds is None:
            return np.array([])
        return _structured_array(ds, fields)

    def annotations_array(self, fields=None):
        """Return annotations as a numpy structured array (colon-keyed columns)."""
        ds = self._dataset("annotations")
        if ds is None:
            return np.array([])
        return _structured_array(ds, fields)

    # -- compatibility helpers (materialize list-of-dicts on demand) -------
    def get_captures(self):
        """Return all captures as a list of dicts (compatibility path)."""
        ds = self._dataset("captures")
        return [] if ds is None else records_from_dataset(ds)

    def get_annotations(self, index=None):
        """Return annotations as a list of dicts (compatibility path).

        If ``index`` is given, return only annotations spanning that sample
        index, matching :meth:`sigmf.sigmffile.SigMFFile.get_annotations`.
        """
        ds = self._dataset("annotations")
        annotations = [] if ds is None else records_from_dataset(ds)
        if index is None:
            return annotations

        result = []
        for annotation in annotations:
            if index < annotation[keys.SAMPLE_START_KEY]:
                continue
            if keys.SAMPLE_COUNT_KEY in annotation:
                if index >= annotation[keys.SAMPLE_START_KEY] + annotation[keys.SAMPLE_COUNT_KEY]:
                    continue
            result.append(annotation)
        return result

    def as_metadata_dict(self):
        """Return the full SigMF metadata dict (global/captures/annotations)."""
        return {
            "global": dict(self._global),
            "captures": self.get_captures(),
            "annotations": self.get_annotations(),
        }

    def to_sigmffile(self, skip_checksum=True):
        """
        Materialize a canonical :class:`sigmf.sigmffile.SigMFFile`.

        This builds the full list-of-dicts metadata (JSON-speed) and returns a
        standard object that supports the complete SigMFFile API, including
        sample reading when ``data_file`` is known.
        """
        from .sigmffile import SigMFFile

        return SigMFFile(
            metadata=self.as_metadata_dict(),
            data_file=self.data_file,
            skip_checksum=skip_checksum,
        )

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        """Close the underlying HDF5 file handle (idempotent)."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ARG002
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort cleanup during GC
            pass


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def open(file_path, data_file=None, global_obj=None):
    """
    Open an HDF5 metadata sidecar directly, reading no JSON at all.

    This is the tightest fast path: when the caller already knows the ``.h5``
    location, no ``.sigmf-meta`` JSON is read. The ``global`` object is read
    from the sidecar's ``/global`` group unless supplied.

    Parameters
    ----------
    file_path : str | PathLike
        Path to the ``.sigmf-meta.h5`` sidecar.
    data_file : str | PathLike, optional
        Path to the associated dataset, if known.
    global_obj : dict, optional
        Pre-known ``global`` object to use instead of reading it from the file.

    Returns
    -------
    SigMFFileHDF5
        A lazy, columnar reader. Close it (or use ``with``) when done.
    """
    return SigMFFileHDF5(open_hdf5(file_path), global_obj=global_obj, data_file=data_file)


def fromfile(meta_path, require_sidecar=False, verify=True, skip_checksum=False):
    """
    Load a Recording, preferring the HDF5 sidecar when available.

    The ``.sigmf-meta`` JSON is read exactly once for discovery: to learn
    whether an ``hdf5-meta:file`` sidecar is declared and to resolve the
    dataset filename. If the sidecar exists, ``h5py`` is installed, and (when
    ``verify``) its stored digest matches the JSON, a lazy
    :class:`SigMFFileHDF5` is returned. Otherwise a standard
    :class:`sigmf.sigmffile.SigMFFile` is returned (the JSON remains
    authoritative).

    Parameters
    ----------
    meta_path : str | PathLike
        Path to the ``.sigmf-meta`` file (with or without extension).
    require_sidecar : bool, default False
        If True, raise :class:`SigMFHDF5Error` when a usable sidecar is not
        available instead of falling back to JSON.
    verify : bool, default True
        If True, compare the sidecar's stored digest against the JSON and fall
        back (with a warning) if they disagree, guarding against a stale
        sidecar.
    skip_checksum : bool, default False
        Passed through to the JSON fallback ``SigMFFile``.

    Returns
    -------
    SigMFFileHDF5 | sigmf.sigmffile.SigMFFile
    """
    from .sigmffile import fromfile as json_fromfile
    from .sigmffile import get_dataset_filename_from_metadata, get_sigmf_filenames

    fns = get_sigmf_filenames(meta_path)
    meta_fn = fns["meta_fn"]

    def _fallback(reason):
        if require_sidecar:
            raise SigMFHDF5Error(f"No usable hdf5-meta sidecar for '{meta_fn}': {reason}")
        return json_fromfile(meta_path, skip_checksum=skip_checksum)

    if not meta_fn.is_file():
        return _fallback("metadata file not found")

    # single JSON read for discovery (builtins.open: this module shadows open())
    with builtins.open(meta_fn, "rb") as fp:
        metadata = json.loads(fp.read().decode("utf-8"))

    sidecar_name = metadata.get("global", {}).get(HDF5_META_FILE_KEY)
    if not sidecar_name:
        return _fallback("no hdf5-meta:file declared")

    sidecar_path = Path(meta_fn).parent / sidecar_name
    if not sidecar_path.is_file():
        return _fallback(f"sidecar '{sidecar_name}' not found")

    try:
        handle = open_hdf5(sidecar_path)
    except (SigMFHDF5Error, OSError) as exc:
        # SigMFHDF5Error: h5py missing. OSError: not a valid/readable HDF5 file.
        return _fallback(str(exc))

    if verify:
        stored = read_source_digest(handle)
        if stored is not None and stored != _metadata_digest(metadata):
            handle.close()
            warnings.warn(f"hdf5-meta sidecar '{sidecar_path}' is stale (digest mismatch); using JSON metadata.")
            return _fallback("stale sidecar")

    data_fn = get_dataset_filename_from_metadata(meta_fn, metadata)
    return SigMFFileHDF5(handle, global_obj=metadata.get("global", {}), data_file=data_fn)


# ---------------------------------------------------------------------------
# command-line interface
# ---------------------------------------------------------------------------
def main(arg_tuple=None):
    """Command-line entry point for generating HDF5 metadata sidecars.

    Reads one or more existing ``.sigmf-meta`` JSON files and writes an
    ``hdf5-meta`` sidecar alongside each, declaring the extension in the JSON
    (unless ``--no-update-json`` is given). Installed as ``sigmf_hdf5``.
    """
    import argparse
    import glob

    from . import __version__ as toolversion

    parser = argparse.ArgumentParser(
        description="Generate an HDF5 metadata sidecar from an existing SigMF .sigmf-meta file.",
        prog="sigmf_hdf5",
    )
    parser.add_argument(
        "path", nargs="+", help="SigMF metadata path(s). Accepts * wildcards; the extension is optional."
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable gzip compression of the columnar datasets.",
    )
    parser.add_argument(
        "--no-update-json",
        action="store_true",
        help="Do not declare the hdf5-meta extension in the .sigmf-meta JSON (sidecar won't be auto-discovered).",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail instead of overwriting an existing sidecar.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Print each sidecar written.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {toolversion}")

    args = parser.parse_args(arg_tuple)

    # resolve possible wildcards
    paths = []
    for path in args.path:
        expanded = glob.glob(path)
        paths += expanded if expanded else [path]

    n_ok = 0
    for path in paths:
        try:
            sidecar = generate_sidecar(
                path,
                compression=None if args.no_compression else "gzip",
                update_json=not args.no_update_json,
                overwrite=not args.no_overwrite,
            )
        except SigMFHDF5Error as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            continue
        n_ok += 1
        if args.verbose:
            print(f"wrote {sidecar}")

    n_total = len(paths)
    if n_ok != n_total:
        print(f"Generated {n_ok} of {n_total} sidecar(s)", file=sys.stderr)
        sys.exit(1)
    if args.verbose:
        print(f"Generated all {n_total} sidecar(s) OK")


if __name__ == "__main__":
    main()

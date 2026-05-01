# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/SigMF
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Access SigMF archives without extracting them."""

import io
import tarfile
import zipfile
from pathlib import Path

from . import __version__
from .archive import (
    SIGMF_ARCHIVE_EXT,
    SIGMF_ARCHIVE_EXTS,
    SIGMF_DATASET_EXT,
    SIGMF_METADATA_EXT,
    _detect_compression,
)
from .error import SigMFFileError
from .hashing import calculate_sha512
from .sigmffile import SigMFFile


class SigMFArchiveReader:
    """
    Access data within SigMF archive (tar, tar.gz, tar.xz, or zip) in-place.

    For uncompressed tar archives opened by path, data is memory-mapped
    directly from the archive file for efficient access. Compressed archives
    and buffer-based reading load data into memory.

    Parameters
    ----------
    name : str | bytes | PathLike, optional
        Path to archive file to access. Recognized extensions:
        .sigmf, .sigmf.gz, .sigmf.xz, .sigmf.zip
    skip_checksum : bool, optional
        Skip dataset checksum calculation.
    map_readonly : bool, optional
        Indicate whether assignments on the numpy.memmap are allowed.
    archive_buffer : buffer, optional
        Alternative buffer to read archive from.
    autoscale : bool, optional
        If dataset is in a fixed-point representation, scale samples from (min, max) to (-1.0, 1.0).

    Raises
    ------
    SigMFFileError
        Archive file does not exist or is improperly formatted.
    ValueError
        If invalid arguments.
    ValidationError
        If metadata is invalid.
    """

    def __init__(self, name=None, skip_checksum=False, map_readonly=True, archive_buffer=None, autoscale=True):
        if name is not None:
            path = Path(name)
            compression = _detect_compression(path)

            # validate extension
            name_str = str(path).lower()
            if not any(name_str.endswith(ext) for ext in SIGMF_ARCHIVE_EXTS):
                raise SigMFFileError(
                    f"Unrecognized archive extension for '{path.name}'. "
                    f"Recognized extensions: {sorted(SIGMF_ARCHIVE_EXTS)}"
                )

            if compression == "zip":
                json_contents, data_buffer, data_size_bytes = self._read_zip(path)
                self._init_from_buffer(
                    json_contents, data_buffer, data_size_bytes, skip_checksum, map_readonly, autoscale
                )
            elif compression is not None:
                # compressed tar (gz, xz) — must decompress to ram
                json_contents, data_buffer, data_size_bytes = self._read_tar(path)
                self._init_from_buffer(
                    json_contents, data_buffer, data_size_bytes, skip_checksum, map_readonly, autoscale
                )
            else:
                # uncompressed tar — memmap directly
                self._init_from_tar_memmap(path, skip_checksum, map_readonly, autoscale)

        elif archive_buffer is not None:
            # try tar first, fall back to zip
            try:
                tar_obj = tarfile.open(fileobj=archive_buffer, mode="r:*")
                json_contents, data_buffer, data_size_bytes = self._read_tar_obj(tar_obj)
                tar_obj.close()
            except tarfile.TarError:
                archive_buffer.seek(0)
                json_contents, data_buffer, data_size_bytes = self._read_zip_fileobj(archive_buffer)
            self._init_from_buffer(json_contents, data_buffer, data_size_bytes, skip_checksum, map_readonly, autoscale)

        else:
            raise ValueError("Either `name` or `archive_buffer` must be not None.")

    def _read_tar_obj(self, tar_obj):
        """Extract metadata and data from an open tar object."""
        json_contents = None
        data_buffer = None
        data_size_bytes = None

        for memb in tar_obj.getmembers():
            if memb.isdir():
                continue
            elif memb.isfile():
                if memb.name.endswith(SIGMF_METADATA_EXT):
                    with tar_obj.extractfile(memb) as fid:
                        json_contents = fid.read()
                elif memb.name.endswith(SIGMF_DATASET_EXT):
                    data_size_bytes = memb.size
                    with tar_obj.extractfile(memb) as fid:
                        data_buffer = io.BytesIO(fid.read())

        if data_buffer is None:
            raise SigMFFileError("No .sigmf-data file found in archive!")
        return json_contents, data_buffer, data_size_bytes

    def _read_tar(self, path):
        """Read a tar archive (possibly compressed) from disk."""
        tar_obj = tarfile.open(path)
        result = self._read_tar_obj(tar_obj)
        tar_obj.close()
        return result

    def _read_zip(self, path):
        """Read a zip archive from disk."""
        with zipfile.ZipFile(path, "r") as zf:
            return self._read_zip_obj(zf)

    def _read_zip_fileobj(self, fileobj):
        """Read a zip archive from a buffer."""
        with zipfile.ZipFile(fileobj, "r") as zf:
            return self._read_zip_obj(zf)

    def _read_zip_obj(self, zf):
        """Extract metadata and data from an open ZipFile object."""
        json_contents = None
        data_buffer = None
        data_size_bytes = None

        for member_name in zf.namelist():
            if member_name.endswith(SIGMF_METADATA_EXT):
                json_contents = zf.read(member_name)
            elif member_name.endswith(SIGMF_DATASET_EXT):
                raw = zf.read(member_name)
                data_size_bytes = len(raw)
                data_buffer = io.BytesIO(raw)

        if data_buffer is None:
            raise SigMFFileError("No .sigmf-data file found in archive!")
        return json_contents, data_buffer, data_size_bytes

    def _init_from_buffer(self, json_contents, data_buffer, data_size_bytes, skip_checksum, map_readonly, autoscale):
        """Initialize sigmffile from in-memory data."""
        self.sigmffile = SigMFFile(metadata=json_contents, autoscale=autoscale)
        self.sigmffile.validate()
        self.sigmffile.set_data_file(
            data_buffer=data_buffer,
            skip_checksum=skip_checksum,
            size_bytes=data_size_bytes,
            map_readonly=map_readonly,
        )
        self.ndim = self.sigmffile.ndim
        self.shape = self.sigmffile.shape

    def _init_from_tar_memmap(self, path, skip_checksum, map_readonly, autoscale):
        """Initialize sigmffile with memmap into uncompressed tar."""
        tar_obj = tarfile.open(path)
        json_contents = None
        data_offset = None
        data_size_bytes = None

        for memb in tar_obj.getmembers():
            if memb.isdir():
                continue
            elif memb.isfile():
                if memb.name.endswith(SIGMF_METADATA_EXT):
                    with tar_obj.extractfile(memb) as fid:
                        json_contents = fid.read()
                elif memb.name.endswith(SIGMF_DATASET_EXT):
                    data_offset = memb.offset_data
                    data_size_bytes = memb.size

        tar_obj.close()

        if data_offset is None:
            raise SigMFFileError("No .sigmf-data file found in archive!")

        self.sigmffile = SigMFFile(metadata=json_contents, autoscale=autoscale)
        self.sigmffile.validate()

        # compute hash of data portion only (not full tar file)
        if not skip_checksum:
            data_hash = calculate_sha512(filename=path, offset=data_offset, size=data_size_bytes)
            old_hash = self.sigmffile.get_global_field(SigMFFile.HASH_KEY)
            if old_hash is not None and old_hash != data_hash:
                raise SigMFFileError("Calculated file hash does not match associated metadata.")
            self.sigmffile.set_global_field(SigMFFile.HASH_KEY, data_hash)

        # memmap directly into the tar file at the data offset
        self.sigmffile.set_data_file(
            data_file=path,
            skip_checksum=True,
            offset=data_offset,
            size_bytes=data_size_bytes,
            map_readonly=map_readonly,
        )
        # set_data_file sets DATASET_KEY for non-.sigmf-data files (NCD),
        # but the tar archive path is not a dataset — clear it
        if SigMFFile.DATASET_KEY in self.sigmffile.get_global_info():
            del self.sigmffile._metadata[SigMFFile.GLOBAL_KEY][SigMFFile.DATASET_KEY]

        self.ndim = self.sigmffile.ndim
        self.shape = self.sigmffile.shape

    def __len__(self):
        return self.sigmffile.__len__()

    def __iter__(self):
        return self.sigmffile.__iter__()

    def __getitem__(self, sli):
        return self.sigmffile.__getitem__(sli)

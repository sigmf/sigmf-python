# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create and extract SigMF archives."""

import io
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path

from .error import SigMFFileError, SigMFFileExistsError
from .keys import (
    SIGMF_ARCHIVE_EXT,
    SIGMF_ARCHIVE_EXTS,
    SIGMF_COLLECTION_EXT,
    SIGMF_COMPRESSED_EXTS,
    SIGMF_DATASET_EXT,
    SIGMF_METADATA_EXT,
)


def _detect_compression(path):
    """Detect compression type from a file path's extension(s).

    Parameters
    ----------
    path : Path
        Path to check.

    Returns
    -------
    str or None
        Compression type ("gz", "xz", "zip") or None for uncompressed.
    """
    name = str(path).lower()
    for comp_type, ext in SIGMF_COMPRESSED_EXTS.items():
        if name.endswith(ext):
            return comp_type
    return None


def _get_archive_basename(path):
    """Get the archive base name (without any sigmf archive extension).

    Parameters
    ----------
    path : Path
        Archive file path.

    Returns
    -------
    str
        Base name without sigmf extension.

    Examples
    --------
    >>> _get_archive_basename(Path("recording.sigmf"))
    'recording'
    >>> _get_archive_basename(Path("recording.sigmf.gz"))
    'recording'
    >>> _get_archive_basename(Path("my.recording.sigmf.zip"))
    'my.recording'
    """
    name = path.name
    # check compound extensions first (longest match)
    for ext in sorted(SIGMF_COMPRESSED_EXTS.values(), key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    if name.endswith(SIGMF_ARCHIVE_EXT):
        return name[: -len(SIGMF_ARCHIVE_EXT)]
    return path.stem


def _get_archive_basename(path):
    """Get the archive base name (without any sigmf archive extension).

    Parameters
    ----------
    path : Path
        Archive file path.

    Returns
    -------
    str
        Base name without sigmf extension.

    Examples
    --------
    >>> _get_archive_basename(Path("recording.sigmf"))
    'recording'
    >>> _get_archive_basename(Path("recording.sigmf.gz"))
    'recording'
    >>> _get_archive_basename(Path("my.recording.sigmf.zip"))
    'my.recording'
    """
    name = path.name
    # check compound extensions first (longest match)
    for ext in sorted(SIGMF_COMPRESSED_EXTS.values(), key=len, reverse=True):
        if name.endswith(ext):
            return name[: -len(ext)]
    if name.endswith(SIGMF_ARCHIVE_EXT):
        return name[: -len(SIGMF_ARCHIVE_EXT)]
    return path.stem


class SigMFArchive:
    """
    Archive a SigMFFile into a tar or zip file, optionally with compression.

    Parameters
    ----------

    sigmffile : SigMFFile
        A SigMFFile object with valid metadata and data_file.

    name : PathLike | str | bytes
        Path to archive file to create.
        If `name` doesn't end in a recognized sigmf archive extension,
        .sigmf will be appended. Recognized extensions:
        .sigmf, .sigmf.gz, .sigmf.xz, .sigmf.zip
        For example: if `name` == "/tmp/archive1", then the
        following archive will be created:
            /tmp/archive1.sigmf
            - archive1/
                - archive1.sigmf-meta
                - archive1.sigmf-data

    fileobj : BufferedWriter
        If `fileobj` is specified, it is used as an alternative to
        a file object opened in binary mode for `name`. It is
        supposed to be at position 0. `name` is not required, but
        if specified will be used to determine the directory and
        file names within the archive. `fileobj` won't be closed.
        For example: if `name` == "archive1" and fileobj is given,
        a tar archive will be written to fileobj with the
        following structure:
            - archive1/
                - archive1.sigmf-meta
                - archive1.sigmf-data

    compression : str, optional
        Compression type: "gz", "xz", "zip", or None (default).
        If None and `name` has a recognized compressed extension,
        compression is auto-detected from the extension.

    overwrite : bool, default False
        If False, raise exception if archive file already exists.

    Raises
    ------
    SigMFFileError
        If `sigmffile` has no data_file set, or if `name` is not writable,
        or if an invalid compression type is given.

    """

    VALID_COMPRESSIONS = {None, "gz", "xz", "zip"}

    def __init__(self, sigmffile, name=None, fileobj=None, compression=None, overwrite=False):
        is_buffer = fileobj is not None
        self.sigmffile = sigmffile
        self.path, arcname, fileobj, compression = self._resolve(name, fileobj, compression, overwrite)

        self._ensure_data_file_set()
        self._validate()

        # prepare temp files with metadata and data
        tmpdir = Path(tempfile.mkdtemp())
        meta_path = tmpdir / (arcname + SIGMF_METADATA_EXT)
        data_path = tmpdir / (arcname + SIGMF_DATASET_EXT)

        with open(meta_path, "w") as handle:
            self.sigmffile.dump(handle)
        if isinstance(self.sigmffile.data_buffer, io.BytesIO):
            with open(data_path, "wb") as handle:
                handle.write(self.sigmffile.data_buffer.getbuffer())
        else:
            shutil.copy(self.sigmffile.data_file, data_path)

        if compression == "zip":
            self._write_zip(fileobj, arcname, tmpdir, meta_path, data_path)
        else:
            self._write_tar(fileobj, arcname, tmpdir, compression)

        if not is_buffer:
            # only close fileobj if we aren't working w/a buffer
            fileobj.close()
        shutil.rmtree(tmpdir)

    def _write_tar(self, fileobj, arcname, tmpdir, compression):
        """Write archive as tar (optionally compressed)."""
        mode = "w" if compression is None else f"w:{compression}"
        tar = tarfile.open(mode=mode, fileobj=fileobj, format=tarfile.PAX_FORMAT)
        tar.add(tmpdir, arcname=arcname, filter=self.chmod)
        tar.close()

    def _write_zip(self, fileobj, arcname, tmpdir, meta_path, data_path):
        """Write archive as zip."""
        with zipfile.ZipFile(fileobj, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            # add data file first (matches tar convention for faster metadata updates)
            zf.write(data_path, arcname=f"{arcname}/{arcname}{SIGMF_DATASET_EXT}")
            zf.write(meta_path, arcname=f"{arcname}/{arcname}{SIGMF_METADATA_EXT}")

    @staticmethod
    def chmod(tarinfo: tarfile.TarInfo):
        """permission filter for writing tar files"""
        if tarinfo.isdir():
            tarinfo.mode = 0o755  # dwrxw-rw-r
        else:
            tarinfo.mode = 0o644  # -wr-r--r--
        return tarinfo

    def _ensure_data_file_set(self):
        if not self.sigmffile.data_file and not isinstance(self.sigmffile.data_buffer, io.BytesIO):
            raise SigMFFileError("No data file in SigMFFile; use `set_data_file` before archiving.")

    def _validate(self):
        self.sigmffile.validate()

    def _resolve(self, name, fileobj, compression, overwrite=False):
        """
        Resolve both (name, fileobj) into (path, arcname, fileobj, compression) given either or both.

        Parameters
        ----------
        name : PathLike | str | bytes | None
            Path to archive file to create.
        fileobj : BufferedWriter | None
            Open file handle object.
        compression : str | None
            Compression type or None.
        overwrite : bool, default False
            If False, raise exception if archive file already exists.

        Returns
        -------
        path : Path
            Path of the archive file.
        arcname : str
            Name of the sigmf object within the archive.
        fileobj : BufferedWriter
            Open file handle object.
        compression : str | None
            Resolved compression type.
        """
        if compression not in self.VALID_COMPRESSIONS:
            raise SigMFFileError(f"Invalid compression type '{compression}'. Must be one of: {self.VALID_COMPRESSIONS}")

        if fileobj:
            try:
                fileobj.write(bytes())
                path = Path(fileobj.name)
                if not name:
                    arcname = _get_archive_basename(path)
                else:
                    arcname = name
            except io.UnsupportedOperation as exc:
                raise SigMFFileError(f"fileobj {fileobj} is not byte-writable.") from exc
            except AttributeError as exc:
                raise SigMFFileError(f"fileobj {fileobj} is invalid.") from exc
        elif name:
            path = Path(name)
            name_str = str(path).lower()

            # auto-detect compression from extension if not explicitly set
            detected = _detect_compression(path)
            if compression is None and detected is not None:
                compression = detected

            # check if path has a recognized archive extension
            has_archive_ext = any(name_str.endswith(ext) for ext in SIGMF_ARCHIVE_EXTS)

            if not has_archive_ext:
                if path.suffix == "":
                    # no extension — append the appropriate one
                    if compression is not None:
                        path = Path(str(path) + SIGMF_COMPRESSED_EXTS[compression])
                    else:
                        path = path.with_suffix(SIGMF_ARCHIVE_EXT)
                else:
                    # has an unrecognized extension
                    raise SigMFFileError(
                        f"Unrecognized archive extension for '{path.name}'. "
                        f"Recognized extensions: {sorted(SIGMF_ARCHIVE_EXTS)}"
                    )
            elif detected is not None and compression is not None and detected != compression:
                raise SigMFFileError(
                    f"Extension implies '{detected}' compression but compression='{compression}' was specified."
                )

            arcname = _get_archive_basename(path)

            if not overwrite and path.exists():
                raise SigMFFileExistsError(path, "Archive file")

            try:
                fileobj = open(path, "wb")
            except (OSError, IOError) as exc:
                raise SigMFFileError(f"Can't open {name} for writing.") from exc
        else:
            raise SigMFFileError("Either `name` or `fileobj` needs to be defined.")

        return path, arcname, fileobj, compression

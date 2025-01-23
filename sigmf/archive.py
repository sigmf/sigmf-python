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
from pathlib import Path

from .error import SigMFFileError

SIGMF_ARCHIVE_EXT = ".sigmf"
SIGMF_METADATA_EXT = ".sigmf-meta"
SIGMF_DATASET_EXT = ".sigmf-data"
SIGMF_COLLECTION_EXT = ".sigmf-collection"


class SigMFArchive:
    """
    Archive a SigMFFile

    A `.sigmf` file must include both valid metadata and data.
    If `self.data_file` is not set or the requested output file
    is not writable, raises `SigMFFileError`.

    Parameters
    ----------

    sigmffile : SigMFFile
        A SigMFFile object with valid metadata and data_file.

    name : PathLike | str | bytes
        Path to archive file to create. If file exists, overwrite.
        If `name` doesn't end in .sigmf, it will be appended.
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
    """

    def __init__(self, sigmffile, name=None, fileobj=None):
        is_buffer = fileobj is not None
        self.sigmffile = sigmffile
        self.path, arcname, fileobj = self._resolve(name, fileobj)

        self._ensure_data_file_set()
        self._validate()

        tar = tarfile.TarFile(mode="w", fileobj=fileobj, format=tarfile.PAX_FORMAT)
        tmpdir = Path(tempfile.mkdtemp())
        meta_path = tmpdir / (arcname + SIGMF_METADATA_EXT)
        data_path = tmpdir / (arcname + SIGMF_DATASET_EXT)

        # write files
        with open(meta_path, "w") as handle:
            self.sigmffile.dump(handle)
        if isinstance(self.sigmffile.data_buffer, io.BytesIO):
            # write data buffer to archive
            self.sigmffile.data_file = data_path
            with open(data_path, "wb") as handle:
                handle.write(self.sigmffile.data_buffer.getbuffer())
        else:
            # copy data to archive
            shutil.copy(self.sigmffile.data_file, data_path)
        tar.add(tmpdir, arcname=arcname, filter=self.chmod)
        # close files & remove tmpdir
        tar.close()
        if not is_buffer:
            # only close fileobj if we aren't working w/a buffer
            fileobj.close()
        shutil.rmtree(tmpdir)

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

    def _resolve(self, name, fileobj):
        """
        Resolve both (name, fileobj) into (path, arcname, fileobj) given either or both.

        Returns
        -------
        path : PathLike
            Path of the archive file.
        arcname : str
            Name of the sigmf object within the archive.
        fileobj : BufferedWriter
            Open file handle object.
        """
        if fileobj:
            try:
                # exception if not byte-writable
                fileobj.write(bytes())
                # exception if no name property of handle
                path = Path(fileobj.name)
                if not name:
                    arcname = path.stem
                else:
                    arcname = name
            except io.UnsupportedOperation as exc:
                raise SigMFFileError(f"fileobj {fileobj} is not byte-writable.") from exc
            except AttributeError as exc:
                raise SigMFFileError(f"fileobj {fileobj} is invalid.") from exc
        elif name:
            path = Path(name)
            # ensure name has correct suffix if it exists
            if path.suffix == "":
                # add extension if none was given
                path = path.with_suffix(SIGMF_ARCHIVE_EXT)
            elif path.suffix != SIGMF_ARCHIVE_EXT:
                # ensure suffix is correct
                raise SigMFFileError(f"Invalid extension ({path.suffix} != {SIGMF_ARCHIVE_EXT}).")
            arcname = path.stem

            try:
                fileobj = open(path, "wb")
            except (OSError, IOError) as exc:
                raise SigMFFileError(f"Can't open {name} for writing.") from exc
        else:
            raise SigMFFileError("Either `name` or `fileobj` needs to be defined.")

        return path, arcname, fileobj

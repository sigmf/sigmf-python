# Copyright: Multiple Authors
#
# This file is part of SigMF. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Create and extract SigMF archives."""

import collections
from io import BytesIO
import os
import tarfile
import tempfile
import time
from typing import BinaryIO, Iterable, Union

import sigmf


from .error import SigMFFileError


SIGMF_ARCHIVE_EXT = ".sigmf"
SIGMF_METADATA_EXT = ".sigmf-meta"
SIGMF_DATASET_EXT = ".sigmf-data"
SIGMF_COLLECTION_EXT = ".sigmf-collection"


class SigMFArchive():
    """Archive one or more `SigMFFile`s.

    A `.sigmf` file must include both valid metadata and data.
    If `self.data_file` is not set or the requested output file
    is not writable, raise `SigMFFileError`.

    Parameters:

      sigmffiles -- A single SigMFFile or an iterable of SigMFFile objects with
                    valid metadata and data_files

      path       -- Path to archive file to create. If file exists, overwrite.
                    If `path` doesn't end in .sigmf, it will be appended. The
                    `self.path` instance variable will be updated upon
                    successful writing of the archive to point to the final
                    archive path.


      fileobj    -- If `fileobj` is specified, it is used as an alternative to
                    a file object opened in binary mode for `path`. If
                    `fileobj` is an open tarfile, it will be appended to. It is
                    supposed to be at position 0. `fileobj` won't be closed. If
                    `fileobj` is given, `path` has no effect.

      pretty     -- If True, pretty print JSON when creating the metadata
                    files in the archive. Defaults to True.
    """
    def __init__(self,
                 sigmffiles: Union["sigmf.sigmffile.SigMFFile",
                                   Iterable["sigmf.sigmffile.SigMFFile"]],
                 path: Union[str, os.PathLike] = None,
                 fileobj: BinaryIO = None,
                 pretty=True):

        if (not path) and (not fileobj):
            raise SigMFFileError("'path' or 'fileobj' required for creating "
                                 "SigMF archive!")

        if isinstance(sigmffiles, sigmf.sigmffile.SigMFFile):
            self.sigmffiles = [sigmffiles]
        elif (hasattr(collections, "Iterable") and
              isinstance(sigmffiles, collections.Iterable)):
            self.sigmffiles = sigmffiles
        elif isinstance(sigmffiles, collections.abc.Iterable):  # python 3.10
            self.sigmffiles = sigmffiles
        else:
            raise SigMFFileError("Unknown type for sigmffiles argument!")

        if path:
            self.path = str(path)
        else:
            self.path = None
        self.fileobj = fileobj

        self._check_input()

        mode = "a" if fileobj is not None else "w"
        sigmf_fileobj = self._get_output_fileobj()
        try:
            sigmf_archive = tarfile.TarFile(mode=mode,
                                            fileobj=sigmf_fileobj,
                                            format=tarfile.PAX_FORMAT)
        except tarfile.ReadError:
            # fileobj doesn't contain any archives yet, so reopen in 'w' mode
            sigmf_archive = tarfile.TarFile(mode='w',
                                            fileobj=sigmf_fileobj,
                                            format=tarfile.PAX_FORMAT)

        def chmod(tarinfo):
            if tarinfo.isdir():
                tarinfo.mode = 0o755  # dwrxw-rw-r
            else:
                tarinfo.mode = 0o644  # -wr-r--r--
            return tarinfo

        for sigmffile in self.sigmffiles:
            self._create_parent_dirs(sigmf_archive, sigmffile.name, chmod)
            file_path = os.path.join(sigmffile.name,
                                     os.path.basename(sigmffile.name))
            sf_md_filename = file_path + SIGMF_METADATA_EXT
            sf_data_filename = file_path + SIGMF_DATASET_EXT
            metadata = sigmffile.dumps(pretty=pretty)
            metadata_tarinfo = tarfile.TarInfo(sf_md_filename)
            metadata_tarinfo.size = len(metadata)
            metadata_tarinfo.mtime = time.time()
            metadata_tarinfo = chmod(metadata_tarinfo)
            metadata_buffer = BytesIO(metadata.encode("utf-8"))
            sigmf_archive.addfile(metadata_tarinfo, fileobj=metadata_buffer)
            data_tarinfo = sigmf_archive.gettarinfo(name=sigmffile.data_file,
                                                    arcname=sf_data_filename)
            data_tarinfo = chmod(data_tarinfo)
            with open(sigmffile.data_file, "rb") as data_file:
                sigmf_archive.addfile(data_tarinfo, fileobj=data_file)

        sigmf_archive.close()
        if not fileobj:
            sigmf_fileobj.close()
        else:
            sigmf_fileobj.seek(0)  # ensure next open can read this as a tar

        self.path = sigmf_archive.name

    def _create_parent_dirs(self, _tarfile, sigmffile_name, set_permission):
        path_components = sigmffile_name.split(os.path.sep)
        current_path = ""
        for path in path_components:
            current_path = os.path.join(current_path, path)
            path_found = False
            for member in _tarfile.getmembers():
                if member.name == current_path:
                    path_found = True
                    break
            if not path_found:
                tarinfo = tarfile.TarInfo(current_path)
                tarinfo.type = tarfile.DIRTYPE
                tarinfo = set_permission(tarinfo)
                _tarfile.addfile(tarinfo)

    def _check_input(self):
        self._ensure_path_has_correct_extension()
        for sigmffile in self.sigmffiles:
            self._ensure_sigmffile_name_set(sigmffile)
            self._ensure_data_file_set(sigmffile)
            self._validate_sigmffile_metadata(sigmffile)

    def _ensure_path_has_correct_extension(self):
        path = self.path
        if path is None:
            return

        has_extension = "." in path
        has_correct_extension = path.endswith(SIGMF_ARCHIVE_EXT)
        if has_extension and not has_correct_extension:
            apparent_ext = os.path.splitext(path)[-1]
            err = "extension {} != {}".format(apparent_ext, SIGMF_ARCHIVE_EXT)
            raise SigMFFileError(err)

        self.path = path if has_correct_extension else path + SIGMF_ARCHIVE_EXT

    @staticmethod
    def _ensure_sigmffile_name_set(sigmffile):
        if not sigmffile.name:
            err = "the `name` attribute must be set to pass to `SigMFArchive`"
            raise SigMFFileError(err)

    @staticmethod
    def _ensure_data_file_set(sigmffile):
        if not sigmffile.data_file:
            err = "no data file - use `set_data_file`"
            raise SigMFFileError(err)

    @staticmethod
    def _validate_sigmffile_metadata(sigmffile):
        sigmffile.validate()

    def _get_archive_name(self):
        if self.fileobj and not self.path:
            pathname = self.fileobj.name
        else:
            pathname = self.path

        filename = os.path.split(pathname)[-1]
        archive_name, archive_ext = os.path.splitext(filename)
        return archive_name

    def _get_output_fileobj(self):
        try:
            fileobj = self._get_open_fileobj()
        except:
            if self.fileobj:
                err = "fileobj {!r} is not byte-writable".format(self.fileobj)
            else:
                err = "can't open {!r} for writing".format(self.path)

            raise SigMFFileError(err)

        return fileobj

    def _get_open_fileobj(self):
        if self.fileobj:
            fileobj = self.fileobj
            fileobj.write(bytes())  # force exception if not byte-writable
        else:
            fileobj = open(self.path, "wb")

        return fileobj

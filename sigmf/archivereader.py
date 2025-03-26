# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/SigMF
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Access SigMF archives without extracting them."""

import io
import tarfile
from pathlib import Path

from . import __version__
from .archive import SIGMF_ARCHIVE_EXT, SIGMF_DATASET_EXT, SIGMF_METADATA_EXT
from .error import SigMFFileError
from .sigmffile import SigMFFile


class SigMFArchiveReader:
    """
    Access data within SigMF archive tarball in-place without extracting.

    Parameters
    ----------
    name : str | bytes | PathLike, optional
        Optional path to archive file to access.
    skip_checksum : bool, optional
        Skip dataset checksum calculation.
    map_readonly : bool, optional
        Indicate whether assignments on the numpy.memmap are allowed.
    archive_buffer : buffer, optional


    Raises
    ------
    SigMFError
        Archive file does not exist or is improperly formatted.
    ValueError
        If invalid arguments.
    ValidationError
        If metadata is invalid.
    """

    def __init__(self, name=None, skip_checksum=False, map_readonly=True, archive_buffer=None):
        if name is not None:
            path = Path(name)
            if path.suffix != SIGMF_ARCHIVE_EXT:
                err = "archive extension != {}".format(SIGMF_ARCHIVE_EXT)
                raise SigMFFileError(err)

            tar_obj = tarfile.open(path)

        elif archive_buffer is not None:
            tar_obj = tarfile.open(fileobj=archive_buffer, mode="r:")

        else:
            raise ValueError("Either `name` or `archive_buffer` must be not None.")

        json_contents = None
        data_offset = None
        data_size_bytes = None

        for memb in tar_obj.getmembers():
            if memb.isdir():  # memb.type == tarfile.DIRTYPE:
                # the directory structure will be reflected in the member name
                continue

            elif memb.isfile():  # memb.type == tarfile.REGTYPE:
                if memb.name.endswith(SIGMF_METADATA_EXT):
                    json_contents = memb.name
                    if data_offset is None:
                        # consider a warnings.warn() here; the datafile should be earlier in the
                        # archive than the metadata, so that updating it (like, adding an annotation)
                        # is fast.
                        pass
                    with tar_obj.extractfile(memb) as memb_fid:
                        json_contents = memb_fid.read()

                elif memb.name.endswith(SIGMF_DATASET_EXT):
                    data_offset = memb.offset_data
                    data_size_bytes = memb.size
                    with tar_obj.extractfile(memb) as memb_fid:
                        data_buffer = io.BytesIO(memb_fid.read())

                else:
                    print(f"A regular file {memb.name} was found but ignored in the archive")
            else:
                print(f"A member of type {memb.type} and name {memb.name} was found but not handled, just FYI.")

        if data_offset is None:
            raise SigMFFileError("No .sigmf-data file found in archive!")

        self.sigmffile = SigMFFile(metadata=json_contents)
        self.sigmffile.validate()

        self.sigmffile.set_data_file(
            data_buffer=data_buffer,
            skip_checksum=skip_checksum,
            size_bytes=data_size_bytes,
            map_readonly=map_readonly,
        )

        self.ndim = self.sigmffile.ndim
        self.shape = self.sigmffile.shape

        tar_obj.close()

    def __len__(self):
        return self.sigmffile.__len__()

    def __iter__(self):
        return self.sigmffile.__iter__()

    def __getitem__(self, sli):
        return self.sigmffile.__getitem__(sli)

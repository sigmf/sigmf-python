# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Hashing Functions"""

import hashlib
from pathlib import Path


def calculate_sha512(filename=None, fileobj=None):
    """
    Calculate SHA512 hash of a dataset for integrity verification.

    The entire recording file should be hashed according to the SigMF specification.

    Parameters
    ----------
    filename : str or Path, optional
        Path to the file to hash. If provided, the file will be opened and hashed.
        Cannot be used together with fileobj.
    fileobj : file-like object, optional
        An open file-like object (e.g., BytesIO) to hash. Must have read() and
        seek() methods. Cannot be used together with filename.

    Returns
    -------
    str
        128 character hex digest (512 bits).

    Raises
    ------
    ValueError
        If neither filename nor fileobj is provided.
    """
    the_hash = hashlib.sha512()
    bytes_read = 0

    if filename is not None:
        fileobj = open(filename, "rb")
        bytes_to_hash = Path(filename).stat().st_size
    elif fileobj is not None:
        current_pos = fileobj.tell()
        # seek to end
        fileobj.seek(0, 2)
        bytes_to_hash = fileobj.tell()
        # reset to original position
        fileobj.seek(current_pos)
    else:
        raise ValueError("Either filename or fileobj must be provided")

    while bytes_read < bytes_to_hash:
        buff = fileobj.read(min(4096, (bytes_to_hash - bytes_read)))
        the_hash.update(buff)
        bytes_read += len(buff)

    if filename is not None:
        fileobj.close()

    return the_hash.hexdigest()

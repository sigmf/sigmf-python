# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Defines SigMF exception classes."""


class SigMFError(Exception):
    """SigMF base exception."""


class SigMFValidationError(SigMFError):
    """Exceptions related to validating SigMF metadata."""


class SigMFAccessError(SigMFError):
    """Exceptions related to accessing the contents of SigMF metadata, notably
    when expected fields are missing or accessing out of bounds captures."""


class SigMFFileError(SigMFError):
    """Exceptions related to reading or writing SigMF files or archives."""


class SigMFFileExistsError(SigMFFileError):
    """Exception raised when a file already exists and overwrite is disabled."""

    def __init__(self, file_path, file_type="File"):
        super().__init__(f"{file_type} {file_path} already exists. Use overwrite=True to overwrite.")
        self.file_path = file_path
        self.file_type = file_type


class SigMFConversionError(SigMFError):
    """Exceptions related to converting to SigMF format."""

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


class SigMFConversionError(SigMFError):
    """Exceptions related to converting to SigMF format."""

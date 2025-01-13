# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# version of this python module
__version__ = "1.2.6"
# matching version of the SigMF specification
__specification__ = "1.2.3"

from . import archive, archivereader, error, schema, sigmffile, utils, validate
from .archive import SigMFArchive
from .archivereader import SigMFArchiveReader
from .sigmffile import SigMFCollection, SigMFFile

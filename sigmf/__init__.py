# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# version of this python module
__version__ = "1.2.1"
# matching version of the SigMF specification
__specification__ = "1.2.0"

from .archive import SigMFArchive
from .sigmffile import SigMFFile, SigMFCollection
from .archivereader import SigMFArchiveReader

from . import archive
from . import archivereader
from . import error
from . import schema
from . import sigmffile
from . import utils
from . import validate

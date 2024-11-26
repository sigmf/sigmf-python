# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for loading non-conforming datasets"""

import os
import numpy as np
import pytest
from sigmf.sigmffile import SigMFFile, fromfile


@pytest.mark.parametrize(
    ["file_path"],
    [
        ["b1.bin"],
        ["./b2.bin"],
        ["test_subdir/b3.bin"],  # fails in the 1.2.3 version
        ["./test_subdir/b4.bin"],  # fails in the 1.2.3 version
    ],
)
def test_load_ncd(file_path: str) -> None:
    dir_path, file_name = os.path.split(file_path)
    file_name_base, file_name_ext = os.path.splitext(file_name)
    if not dir_path:
        dir_path = "."  # sets the correct path in the case file is only a filename
    meta_file_path = f"{dir_path}/{file_name_base}.sigmf-meta"

    # create dir
    try:
        os.makedirs(dir_path)
    except FileExistsError:
        pass

    # create dataset
    np.arange(10, dtype=np.int16).tofile(file_path)

    # create metadata file
    metadata = {
        SigMFFile.GLOBAL_KEY: {
            SigMFFile.DATATYPE_KEY: "ri16_le",
            SigMFFile.DATASET_KEY: file_name,
        },
        SigMFFile.CAPTURE_KEY: [
            {
            SigMFFile.START_INDEX_KEY: 0,
            }
        ],
        SigMFFile.ANNOTATION_KEY: [],
    }
    meta_file = SigMFFile(metadata=metadata, data_file=file_path)
    meta_file.tofile(meta_file_path)

    # load dataset
    data = fromfile(meta_file_path)

    assert np.array_equal(
        np.arange(10, dtype=np.int16),
        data.read_samples(autoscale=False),
    )

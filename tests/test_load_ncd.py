# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for loading non-conforming datasets."""

from typing import cast
import os
import numpy as np
import pytest
from sigmf.sigmffile import SigMFFile, fromfile


@pytest.mark.parametrize(
    ["index", "dir_path"],
    enumerate(
        [
            "",
            "./",
            "test_subdir/",
            "./test_subdir/",
        ]
    ),
)
def test_load_ncd(index: int, dir_path: str) -> None:
    """Unit test - loading non-conforming dataset."""
    data_file_name = f"data{index}.bin"
    meta_file_name = f"data{index}.sigmf-meta"
    data_file_path = f"{dir_path}{data_file_name}"
    meta_file_path = f"{dir_path}{meta_file_name}"

    # create dir if necessary
    try:
        if dir_path:
            os.makedirs(dir_path)
    except FileExistsError:
        pass

    # create data file
    data_in = np.arange(10, dtype=np.int16)
    data_in.tofile(data_file_path)

    # create metadata file
    metadata = {
        SigMFFile.GLOBAL_KEY: {
            SigMFFile.DATATYPE_KEY: "ri16_le",
            SigMFFile.DATASET_KEY: data_file_name,
        },
        SigMFFile.CAPTURE_KEY: [
            {
                SigMFFile.START_INDEX_KEY: 0,
            }
        ],
        SigMFFile.ANNOTATION_KEY: [],
    }
    meta_file = SigMFFile(metadata=metadata, data_file=data_file_path)
    meta_file.tofile(meta_file_path)

    # load dataset
    dataset = cast(SigMFFile, fromfile(meta_file_path))
    data_out = dataset.read_samples(autoscale=False)

    assert np.array_equal(data_in, data_out)

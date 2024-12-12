# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""Tests for path handling for collections."""

from typing import cast
import os
import numpy as np
import pytest
from sigmf.sigmffile import SigMFFile, SigMFCollection, fromfile


@pytest.mark.parametrize(
    ["index", "dir_path"],
    enumerate(
        [
            "",
            ".",
            "./",
            "test_subdir",
            "test_subdir/",
            "./test_subdir",
            "./test_subdir/",
        ]
    ),
)
def test_load_collection(index: int, dir_path: str) -> None:
    """Unit test - path handling for collections."""
    data_file1_name = f"data{index}_1.sigmf-data"
    data_file2_name = f"data{index}_2.sigmf-data"
    meta_file1_name = f"data{index}_1.sigmf-meta"
    meta_file2_name = f"data{index}_2.sigmf-meta"
    collection_file_name = f"collection{index}.sigmf-collection"
    data_file1_path = os.path.join(dir_path, data_file1_name)
    data_file2_path = os.path.join(dir_path, data_file2_name)
    meta_file1_path = os.path.join(dir_path, meta_file1_name)
    meta_file2_path = os.path.join(dir_path, meta_file2_name)
    collection_file_path = os.path.join(dir_path, collection_file_name)

    # create dir if necessary
    try:
        if dir_path:
            os.makedirs(dir_path)
    except FileExistsError:
        pass

    # create data files
    data_in1 = np.arange(10, dtype=np.int16)
    data_in2 = np.arange(20, dtype=np.float32)
    data_in1.tofile(data_file1_path)
    data_in2.tofile(data_file2_path)

    # create metadata files
    metadata1 = {
        SigMFFile.GLOBAL_KEY: {
            SigMFFile.DATATYPE_KEY: "ri16_le",
        },
        SigMFFile.CAPTURE_KEY: [
            {
                SigMFFile.START_INDEX_KEY: 0,
            }
        ],
        SigMFFile.ANNOTATION_KEY: [],
    }
    metadata2 = {
        SigMFFile.GLOBAL_KEY: {
            SigMFFile.DATATYPE_KEY: "rf32_le",
        },
        SigMFFile.CAPTURE_KEY: [
            {
                SigMFFile.START_INDEX_KEY: 0,
            }
        ],
        SigMFFile.ANNOTATION_KEY: [],
    }
    meta_file1 = SigMFFile(metadata=metadata1, data_file=data_file1_path)
    meta_file2 = SigMFFile(metadata=metadata2, data_file=data_file2_path)
    meta_file1.tofile(meta_file1_path)
    meta_file2.tofile(meta_file2_path)

    # create collection
    collection = SigMFCollection(
        metafiles=[meta_file1_name, meta_file2_name],
        collection_path=dir_path,
    )
    collection.tofile(collection_file_path)

    # load collection
    datasets = cast(SigMFCollection, fromfile(collection_file_path))
    dataset1 = cast(SigMFFile, datasets.get_SigMFFile(stream_index=0))
    dataset2 = cast(SigMFFile, datasets.get_SigMFFile(stream_index=1))
    data_out1 = dataset1.read_samples(autoscale=False)
    data_out2 = dataset2.read_samples(autoscale=False)

    assert np.array_equal(data_in1, data_out1)
    assert np.array_equal(data_in2, data_out2)

import os
import shutil
import tempfile

import numpy as np

from sigmf import SigMFFile, SigMFArchiveReader
from sigmf.archive import SIGMF_METADATA_EXT, SigMFArchive


def test_access_data_without_untar(test_sigmffile):
    global_info = {
            "core:author": "Glen M",
            "core:datatype": "ri16_le",
            "core:license": "https://creativecommons.org/licenses/by-sa/4.0/",
            "core:num_channels": 2,
            "core:sample_rate": 48000,
            "core:version": "1.0.0"
        }
    capture_info = {
            "core:datetime": "2021-06-18T23:17:51.163959Z",
            "core:sample_start": 0
        }

    NUM_ROWS = 5

    for dt in "ri16_le", "ci16_le", "rf32_le", "rf64_le", "cf32_le", "cf64_le":
        global_info["core:datatype"] = dt
        for num_chan in 1,3:
            global_info["core:num_channels"] = num_chan
            base_filename = dt + '_' + str(num_chan)
            archive_filename = base_filename + '.sigmf'

            a = np.arange(NUM_ROWS * num_chan * (2 if 'c' in dt else 1))
            if 'i16' in dt:
                b = a.astype(np.int16)
            elif 'f32' in dt:
                b = a.astype(np.float32)
            elif 'f64' in dt:
                b = a.astype(np.float64)
            else:
                raise ValueError('whoops')

            test_sigmffile.data_file = None
            with tempfile.NamedTemporaryFile() as temp:
                b.tofile(temp.name)
                meta = SigMFFile("test",
                                 data_file=temp.name,
                                 global_info=global_info)
                meta.add_capture(0, metadata=capture_info)
                meta.tofile(archive_filename, toarchive=True)

                archi = SigMFArchiveReader(archive_filename, skip_checksum=True)


def test_extract_single_recording(test_sigmffile):
    with tempfile.NamedTemporaryFile() as tf:
        expected_sigmffile = test_sigmffile
        arch = SigMFArchive(expected_sigmffile, path=tf.name)
        reader = SigMFArchiveReader(arch.path)
        assert len(reader) == 1
        actual_sigmffile = reader[0]
        assert expected_sigmffile == actual_sigmffile


def test_extract_multi_recording(test_sigmffile, test_alternate_sigmffile):
    with tempfile.NamedTemporaryFile() as tf:
        # Create a multi-recording archive
        expected_sigmffiles = [test_sigmffile, test_alternate_sigmffile]
        arch = SigMFArchive(expected_sigmffiles, path=tf.name)
        reader = SigMFArchiveReader(arch.path)
        assert len(reader) == 2
        for expected in expected_sigmffiles:
            assert expected in reader.sigmffiles


def test_archivereader_different_folder(test_sigmffile,
                                        test_alternate_sigmffile):
    try:
        os.makedirs("folder1", exist_ok=True)
        test_sigmffile.name = os.path.join("folder1", "test1")
        os.makedirs("folder2", exist_ok=True)
        test_alternate_sigmffile.name = os.path.join("folder2", "test2")
        meta1_filepath = test_sigmffile.name + SIGMF_METADATA_EXT
        with open(meta1_filepath, "w") as meta_fd:
            test_sigmffile.dump(meta_fd)
        meta2_filepath = test_alternate_sigmffile.name + SIGMF_METADATA_EXT
        with open(meta2_filepath, "w") as meta_fd:
            test_alternate_sigmffile.dump(meta_fd)

        os.makedirs("archive_folder", exist_ok=True)
        archive_path = os.path.join("archive_folder", "test_archive.sigmf")
        input_sigmffiles = [test_sigmffile, test_alternate_sigmffile]
        arch = SigMFArchive(input_sigmffiles, path=archive_path)
        reader = SigMFArchiveReader(arch.path)
        assert len(reader) == 2  # number of SigMFFiles
        for actual_sigmffile in reader:
            assert actual_sigmffile in input_sigmffiles
    finally:
        if os.path.exists(meta1_filepath):
            os.remove(meta1_filepath)
        if os.path.exists(meta2_filepath):
            os.remove(meta2_filepath)
        if os.path.exists(archive_path):
            os.remove(archive_path)
        if os.path.exists("folder1"):
            shutil.rmtree("folder1")
        if os.path.exists("folder2"):
            shutil.rmtree("folder2")
        if os.path.exists("archive_folder"):
            shutil.rmtree("archive_folder")


def test_archivereader_same_folder(test_sigmffile,
                                   test_alternate_sigmffile):
    try:
        os.makedirs("folder1", exist_ok=True)
        test_sigmffile.name = os.path.join("folder1", "test1")
        test_alternate_sigmffile.name = os.path.join("folder1", "test2")
        meta1_filepath = test_sigmffile.name + SIGMF_METADATA_EXT
        with open(meta1_filepath, "w") as meta_fd:
            test_sigmffile.dump(meta_fd)
        meta2_filepath = test_alternate_sigmffile.name + SIGMF_METADATA_EXT
        with open(meta2_filepath, "w") as meta_fd:
            test_alternate_sigmffile.dump(meta_fd)
        archive_path = os.path.join("folder1", "test_archive.sigmf")
        input_sigmffiles = [test_sigmffile, test_alternate_sigmffile]
        arch = SigMFArchive(input_sigmffiles, path=archive_path)
        reader = SigMFArchiveReader(arch.path)
        assert len(reader) == 2  # number of SigMFFiles
        for actual_sigmffile in reader:
            assert actual_sigmffile in input_sigmffiles
    finally:
        if os.path.exists(meta1_filepath):
            os.remove(meta1_filepath)
        if os.path.exists(meta2_filepath):
            os.remove(meta2_filepath)
        if os.path.exists(archive_path):
            os.remove(archive_path)
        if os.path.exists("folder1"):
            shutil.rmtree("folder1")

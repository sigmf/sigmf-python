import tempfile
import unittest
import json

from pathlib import Path

from sigmf import SigMFFile, __specification__, __version__
from sigmf.error import SigMFFileError
from sigmf.component.pydantic_metadata import SigMFMetaFileSchema, SigMFGlobalInfo

from .testdata import TEST_FLOAT32_DATA, TEST_METADATA


# just contains global info.
TEST_METADATA1= {
    "global": {
        "core:datatype": "cf32_le"
    },
}

# correct case
TEST_METADATA2 = {
    "annotations": [{"core:sample_count": 16, "core:sample_start": 0}],
    "captures": [{"core:sample_start": 0}],
    "global": {
        "core:datatype": "rf32_le",
        "core:sha512": "f4984219b318894fa7144519185d1ae81ea721c6113243a52b51e444512a39d74cf41a4cec3c5d000bd7277cc71232c04d7a946717497e18619bdbe94bfeadd6",
        "core:num_channels": 1,
        "core:version": __specification__,
        "core:license": "https://creativecommons.org/licenses/by-sa/4.0/"
    },
}

# incorrect case (large < smaller)
TEST_METADATA_FREQ_T = {
    "annotations": [{"core:sample_count": 16, "core:sample_start": 0, 
                     "core:freq_lower_edge": 2e6, "core:freq_upper_edge": 1e6}],
    "captures": [{"core:sample_start": 0}],
    "global": {
        "core:datatype": "rf32_le",
        "core:version": __specification__,
    },
}


def _create_test_file(p: Path, 
                      src=TEST_METADATA2, 
                      include_dataset: bool = True):
    """Writes a SigMF data and metafile to a temporary directory."""
    # write to file.
    TEST_FLOAT32_DATA.tofile(p.with_suffix(".sigmf-data"))
    metafile = SigMFMetaFileSchema(**src)
    
    if include_dataset:
        # add 'dataset' key
        metafile.global_info.dataset = p.with_suffix(".sigmf-data")

    # write meta fileto disk
    with open(p.with_suffix(".sigmf-meta"), "wt") as js_file:
        js_file.write(metafile.model_dump_json(by_alias=True, exclude_none=True))

    # write schema to disk
    with open(p.with_suffix(".json"), "wt") as js_file:
        schema_json = metafile.model_json_schema(by_alias=True)
        json.dump(schema_json, js_file)


class DefaultCases(unittest.TestCase):
    """Cases of default loading of the SigMF pydantic object."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        _create_test_file(self.tmp_path, include_dataset=False)
    
    def test_cov_serialization1(self):
        """Tests the field and model serialization."""
        smfs = SigMFMetaFileSchema.from_file(self.tmp_path)
        # ensure core:dataset checks are validated
        smfs.global_info.dataset = Path(self.tmp_path.with_suffix(".sigmf-data"))
        # export
        smfs.model_dump_json(by_alias=True, exclude_none=True)

    def test_cov_validation1(self):
        """Tests the validation of sorting coverage."""
        # uses just global_info.
        SigMFMetaFileSchema(**TEST_METADATA1)

    def test_change_dataset_extension_auto(self):
        """Changes the core:dataset to .sigmf-data"""
        SigMFGlobalInfo(**{
            "core:datatype": "rf32_le",
            "core:version": __specification__,
            "core:dataset": str(self.tmp_path.with_suffix(".sigmf-meta"))
        })


class WarningCases(unittest.TestCase):
    """Cases where the validator should throws a warning."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir.name)
        TEST_FLOAT32_DATA.tofile(self.tmp_path.with_suffix(".sigmf-data"))

    def test_warn_freq_upper_lower_range(self):
        """Checks that upper_edge >= lower_range"""
        with self.assertWarns(Warning):
            SigMFMetaFileSchema(**TEST_METADATA_FREQ_T)

    def test_dataset_metadata_only_exclusive(self):
        """Checks that metadata_only and dataset are either set."""
        with self.assertWarns(Warning):
            SigMFGlobalInfo(**{
                "core:datatype": "rf32_le",
                "core:version": __specification__,
                "core:dataset": str(self.tmp_path.with_suffix(".sigmf-data")),
                "core:metadata_only": False
            })

    def test_none_collection(self):
        """Field validates the collection attribute"""
        with self.assertWarns(Warning):
            SigMFGlobalInfo(**{
                "core:datatype": "rf32_le",
                "core:version": __specification__,
                "core:collection": "hello there"
            })

    def test_no_datafile_present(self):
        """Validates whether a warning is raised for if the datafile is present (or not.)"""
        with self.assertWarns(Warning):
            SigMFGlobalInfo(**{
                "core:datatype": "rf32_le",
                "core:dataset": "file_no_exist.sigmf-data"
            })


class FailingCases(unittest.TestCase):
    """Cases where the validator should throw an exception."""

    def test_file_does_not_exist(self):
        """Checks whether the .sigmf-meta file exists."""
        with self.assertRaises(SigMFFileError):
            SigMFMetaFileSchema.from_file(Path("./no_exist"))

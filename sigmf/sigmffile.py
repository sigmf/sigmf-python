# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""SigMFFile Object"""

import codecs
import io
import json
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import __specification__, __version__, hashing, schema, validate
from .archive import (
    SIGMF_ARCHIVE_EXT,
    SIGMF_COLLECTION_EXT,
    SIGMF_DATASET_EXT,
    SIGMF_METADATA_EXT,
    SigMFArchive,
)
from .error import SigMFAccessError, SigMFConversionError, SigMFError, SigMFFileError
from .utils import dict_merge, get_magic_bytes


class SigMFMetafile:
    VALID_KEYS = {}

    def __init__(self):
        self.version = None
        self.schema = None
        self._metadata = None
        self.shape = None

    def __str__(self):
        return self.dumps()

    def __repr__(self):
        return f"SigMFFile({self})"

    def __iter__(self):
        """special method to iterate through samples"""
        self.iter_position = 0
        return self

    def ordered_metadata(self):
        """
        Get a nicer representation of _metadata. Will sort keys, but put the
        top-level fields 'global', 'captures', 'annotations' in front.

        Returns
        -------
        ordered_meta : OrderedDict
            Cleaner representation of _metadata with top-level keys correctly
            ordered and the rest of the keys sorted.
        """
        ordered_meta = OrderedDict()
        for top_key in self.VALID_KEYS.keys():
            if top_key not in self._metadata:
                raise SigMFAccessError("key '{}' is not a VALID KEY for metadata".format(top_key))
            ordered_meta[top_key] = json.loads(json.dumps(self._metadata[top_key], sort_keys=True))
        # If there are other top-level keys, they go later
        # TODO: sort potential `other` top-level keys
        for oth_key, oth_val in self._metadata.items():
            if oth_key not in self.VALID_KEYS.keys():
                ordered_meta[oth_key] = json.loads(json.dumps(oth_val, sort_keys=True))
        return ordered_meta

    def dump(self, filep, pretty=True):
        """
        Write metadata to a file.

        Parameters
        ----------
        filep : object
            File pointer or something that json.dump() can handle.
        pretty : bool, default True
            When True will write more human-readable output, otherwise will be flat JSON.
        """
        json.dump(
            self.ordered_metadata(),
            filep,
            indent=4 if pretty else None,
            separators=(",", ": ") if pretty else None,
        )

    def dumps(self, pretty=True):
        """
        Get a string representation of the metadata.

        Parameters
        ----------
        pretty : bool, default True
            When True will write more human-readable output, otherwise will be flat JSON.

        Returns
        -------
        string
            String representation of the metadata using json formatter.
        """
        return json.dumps(
            self.ordered_metadata(),
            indent=4 if pretty else None,
            separators=(",", ": ") if pretty else None,
        )


class SigMFFile(SigMFMetafile):
    START_INDEX_KEY = "core:sample_start"
    LENGTH_INDEX_KEY = "core:sample_count"
    GLOBAL_INDEX_KEY = "core:global_index"
    START_OFFSET_KEY = "core:offset"
    NUM_CHANNELS_KEY = "core:num_channels"
    HASH_KEY = "core:sha512"
    VERSION_KEY = "core:version"
    DATATYPE_KEY = "core:datatype"
    FREQUENCY_KEY = "core:frequency"
    HEADER_BYTES_KEY = "core:header_bytes"
    FLO_KEY = "core:freq_lower_edge"
    FHI_KEY = "core:freq_upper_edge"
    SAMPLE_RATE_KEY = "core:sample_rate"
    COMMENT_KEY = "core:comment"
    DESCRIPTION_KEY = "core:description"
    AUTHOR_KEY = "core:author"
    META_DOI_KEY = "core:meta_doi"
    DATA_DOI_KEY = "core:data_doi"
    GENERATOR_KEY = "core:generator"
    LABEL_KEY = "core:label"
    RECORDER_KEY = "core:recorder"
    LICENSE_KEY = "core:license"
    HW_KEY = "core:hw"
    DATASET_KEY = "core:dataset"
    TRAILING_BYTES_KEY = "core:trailing_bytes"
    METADATA_ONLY_KEY = "core:metadata_only"
    EXTENSIONS_KEY = "core:extensions"
    DATETIME_KEY = "core:datetime"
    LAT_KEY = "core:latitude"
    LON_KEY = "core:longitude"
    UUID_KEY = "core:uuid"
    GEOLOCATION_KEY = "core:geolocation"
    COLLECTION_KEY = "core:collection"
    GLOBAL_KEY = "global"
    CAPTURE_KEY = "captures"
    ANNOTATION_KEY = "annotations"
    VALID_GLOBAL_KEYS = [
        AUTHOR_KEY,
        COLLECTION_KEY,
        DATASET_KEY,
        DATATYPE_KEY,
        DATA_DOI_KEY,
        DESCRIPTION_KEY,
        EXTENSIONS_KEY,
        GEOLOCATION_KEY,
        HASH_KEY,
        HW_KEY,
        LICENSE_KEY,
        META_DOI_KEY,
        METADATA_ONLY_KEY,
        NUM_CHANNELS_KEY,
        RECORDER_KEY,
        SAMPLE_RATE_KEY,
        START_OFFSET_KEY,
        TRAILING_BYTES_KEY,
        VERSION_KEY,
    ]
    VALID_CAPTURE_KEYS = [DATETIME_KEY, FREQUENCY_KEY, HEADER_BYTES_KEY, GLOBAL_INDEX_KEY, START_INDEX_KEY]
    VALID_ANNOTATION_KEYS = [
        COMMENT_KEY,
        FHI_KEY,
        FLO_KEY,
        GENERATOR_KEY,
        LABEL_KEY,
        LAT_KEY,
        LENGTH_INDEX_KEY,
        LON_KEY,
        START_INDEX_KEY,
        UUID_KEY,
    ]
    VALID_KEYS = {GLOBAL_KEY: VALID_GLOBAL_KEYS, CAPTURE_KEY: VALID_CAPTURE_KEYS, ANNOTATION_KEY: VALID_ANNOTATION_KEYS}

    def __init__(
        self, metadata=None, data_file=None, global_info=None, skip_checksum=False, map_readonly=True, autoscale=True
    ):
        """
        API for SigMF I/O

        Parameters
        ----------
        metadata: str or dict, optional
            Metadata for associated dataset.
        data_file: str, optional
            Path to associated dataset.
        global_info: dict, optional
            Set global field shortcut if creating new object.
        skip_checksum: bool, default False
            When True will skip calculating hash on data_file (if present) to check against metadata.
        map_readonly: bool, default True
            Indicates whether assignments on the numpy.memmap are allowed.
        autoscale: bool, default True
            If dataset is in a fixed-point representation, scale samples from (min, max) to (-1.0, 1.0)
            for all sample reading operations including slicing.
        """
        super().__init__()
        self.data_file = None
        self.data_buffer = None
        self.sample_count = 0
        self._memmap = None
        self.is_complex_data = False  # numpy.iscomplexobj(self._memmap) is not adequate for fixed-point complex case
        self.autoscale = autoscale

        self.set_metadata(metadata)
        if global_info is not None:
            self.set_global_info(global_info)
        if data_file is not None:
            offset = self._get_ncd_offset()
            self.set_data_file(data_file, skip_checksum=skip_checksum, map_readonly=map_readonly, offset=offset)

    def __len__(self):
        return self._memmap.shape[0]

    def __eq__(self, other):
        """
        Define equality between two `SigMFFile`s.

        Rely on the checksum value in the metadata to decide whether `data_file` is the same since the path of the
        dataset is immaterial to equivalency.
        """
        if isinstance(other, SigMFFile):
            return self._metadata == other._metadata
        return False

    def __getattr__(self, name):
        """
        Enable dynamic attribute access for core global metadata fields.

        Allows convenient access to core metadata fields using attribute notation:
        - `sigmf_file.sample_rate` returns `sigmf_file._metadata["global"]["core:sample_rate"]
        - `sigmf_file.author` returns `sigmf_file._metadata["global"]["core:author"]

        Parameters
        ----------
        name : str
            Attribute name corresponding to a core field (without "core:" prefix).

        Returns
        -------
        value
            The value of the core field from global metadata, or None if not set.

        Raises
        ------
        SigMFAccessError
            If the attribute name doesn't correspond to a valid core global field.
        """
        # iterate through valid global keys to find matching core field
        for key in self.VALID_GLOBAL_KEYS:
            if key.startswith("core:") and key[5:] == name:
                field_value = self.get_global_field(key)
                if field_value is None:
                    raise SigMFAccessError(f"Core field '{key}' does not exist in global metadata")
                return field_value

        # if we get here, the attribute doesn't correspond to a core field
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """
        Enable dynamic attribute setting for core global metadata fields.

        Allows convenient setting of core metadata fields using attribute notation:
        - `sigmf_file.sample_rate = 1000000` sets `sigmf_file._metadata["global"]["core:sample_rate"]
        - `sigmf_file.author = "jane.doe@domain.org"` sets `sigmf_file._metadata["global"]["core:author"]

        Parameters
        ----------
        name : str
            Attribute name. If it corresponds to a core field (without "core:" prefix),
            the value will be set in global metadata. Otherwise, normal attribute setting occurs.
        value
            The value to set for the field.
        """
        # handle regular instance attributes, existing properties, or during initialization
        if (
            name.startswith("_")
            or hasattr(type(self), name)
            or not hasattr(self, "_metadata")
            or self._metadata is None
        ):
            super().__setattr__(name, value)
            return

        # check if this corresponds to a core global field
        for key in self.VALID_GLOBAL_KEYS:
            if key.startswith("core:") and key[5:] == name:
                self.set_global_field(key, value)
                return

        # fall back to normal attribute setting for non-core attributes
        super().__setattr__(name, value)

    def __next__(self):
        """get next batch of samples"""
        if self.iter_position < len(self):
            # normal batch
            value = self.read_samples(start_index=self.iter_position, count=1)
            self.iter_position += 1
            return value

        else:
            # no more data
            raise StopIteration

    def __getitem__(self, sli):
        """
        Enable slicing and indexing into the dataset samples.

        Should match behavior of ndarray.__getitem__() and apply autoscaling similar to read_samples().
        """
        mem = self._memmap[sli]

        # apply autoscaling for fixed-point data when autoscale=True
        if self.autoscale:
            dtype = dtype_info(self.get_global_field(self.DATATYPE_KEY))
            if dtype["is_fixedpoint"]:
                # extract scaling parameters
                is_unsigned_data = dtype["is_unsigned"]
                component_size = dtype["component_size"]

                # convert to float and apply scaling
                if self.is_complex_data:
                    # for complex data, mem is shaped (..., 2) where last dim is [real, imag]
                    real_part = mem[..., 0].astype(np.float32)
                    imag_part = mem[..., 1].astype(np.float32)

                    # apply scaling to both parts
                    if is_unsigned_data:
                        real_part -= 2 ** (component_size * 8 - 1)
                        imag_part -= 2 ** (component_size * 8 - 1)
                    real_part *= 2 ** -(component_size * 8 - 1)
                    imag_part *= 2 ** -(component_size * 8 - 1)

                    # combine into complex numbers
                    data = real_part + 1.0j * imag_part
                else:
                    # for real data, direct scaling
                    data = mem.astype(np.float32)
                    if is_unsigned_data:
                        data -= 2 ** (component_size * 8 - 1)
                    data *= 2 ** -(component_size * 8 - 1)

                return data

        # handle complex data type conversion if _return_type is set (no autoscaling was applied)
        if self._return_type is not None:
            if self._memmap.ndim == 2:
                # num_channels == 1
                ray = mem[:, 0].astype(self._return_type) + 1.0j * mem[:, 1].astype(self._return_type)
            elif self._memmap.ndim == 3:
                # num_channels > 1
                ray = mem[:, :, 0].astype(self._return_type) + 1.0j * mem[:, :, 1].astype(self._return_type)
            else:
                raise ValueError("unhandled ndim in SigMFFile.__getitem__(); this shouldn't happen")
            return ray[0] if isinstance(sli, int) else ray  # return element instead of 1-element array

        # return raw data (no autoscaling, no complex conversion needed)
        return mem

    def get_num_channels(self):
        """Return integer number of channels."""
        warnings.warn(
            "get_num_channels() is deprecated and will be removed in a future version of sigmf. "
            "Use the 'num_channels' attribute instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_channels

    def _is_conforming_dataset(self):
        """
        The dataset is non-conforming if the datafile contains non-sample bytes
        which means global trailing_bytes field is zero or not set, all captures
        `header_bytes` fields are zero or not set. Because we do not necessarily
        know the filename no means of verifying the meta/data filename roots
        match, but this will also check that a data file exists.

        Returns
        -------
        `True` if the dataset is conforming to SigMF, `False` otherwise
        """
        if self.get_global_field(self.TRAILING_BYTES_KEY, 0):
            return False
        for capture in self.get_captures():
            # check for any non-zero `header_bytes` fields in captures segments
            if capture.get(self.HEADER_BYTES_KEY, 0):
                return False
        if self.data_file is not None and not self.data_file.is_file:
            return False
        # if we get here, the file exists and is conforming
        return True

    def _get_ncd_offset(self):
        """
        Detect Non-Conforming Dataset files and return the appropriate header offset.

        For NCD files that reference external non-SigMF files (e.g., WAV), the
        core:header_bytes field indicates how many bytes to skip to reach the
        actual sample data.

        Returns
        -------
        int
            Byte offset to apply when reading the dataset file. 0 for conforming datasets.
        """
        if self._is_conforming_dataset():
            return 0

        # check if this is an NCD with core:dataset and header_bytes
        captures = self.get_captures()
        dataset_field = self.get_global_field(self.DATASET_KEY)
        if dataset_field and captures and self.HEADER_BYTES_KEY in captures[0]:
            return captures[0][self.HEADER_BYTES_KEY]

        return 0

    def get_schema(self):
        """
        Return a schema object valid for the current metadata
        """
        current_metadata_version = self.get_global_info().get(self.VERSION_KEY)
        if self.version != current_metadata_version or self.schema is None:
            self.version = current_metadata_version
            self.schema = schema.get_schema(self.version)
        if not isinstance(self.schema, dict):
            raise SigMFError("SigMF schema expects a dict (key, value pairs)")
        return self.schema

    def set_metadata(self, metadata):
        """
        Read provided metadata as either None (empty), string, bytes, or dictionary.
        """
        if metadata is None:
            # Create empty
            self._metadata = {self.GLOBAL_KEY: {}, self.CAPTURE_KEY: [], self.ANNOTATION_KEY: []}
        elif isinstance(metadata, dict):
            self._metadata = metadata
        elif isinstance(metadata, (str, bytes)):
            self._metadata = json.loads(metadata)
        else:
            raise SigMFError("Unable to interpret provided metadata.")

        # ensure fields required for parsing are present or use defaults
        if self.get_global_field(self.NUM_CHANNELS_KEY) is None:
            self.set_global_field(self.NUM_CHANNELS_KEY, 1)
        if self.get_global_field(self.START_OFFSET_KEY) is None:
            self.set_global_field(self.START_OFFSET_KEY, 0)

        # set version to current implementation
        self.set_global_field(self.VERSION_KEY, __specification__)

    def set_global_info(self, new_global):
        """
        Recursively override existing global metadata with new global metadata.
        """
        self._metadata[self.GLOBAL_KEY] = dict_merge(self._metadata[self.GLOBAL_KEY], new_global)

    def get_global_info(self):
        """
        Returns a dictionary with all the global info.
        """
        try:
            return self._metadata.get(self.GLOBAL_KEY, {})
        except AttributeError:
            return {}

    def set_global_field(self, key, value):
        """
        Inserts a value into the global field.
        """
        self._metadata[self.GLOBAL_KEY][key] = value

    def get_global_field(self, key, default=None):
        """
        Return a field from the global info, or default if the field is not set.
        """
        return self._metadata[self.GLOBAL_KEY].get(key, default)

    def add_capture(self, start_index, metadata=None):
        """
        Insert capture info for sample starting at start_index.
        If there is already capture info for this index, metadata will be merged
        with the existing metadata, overwriting keys if they were previously set.
        """
        if start_index < self.offset:
            raise SigMFAccessError("Capture start_index cannot be less than dataset start offset.")
        capture_list = self._metadata[self.CAPTURE_KEY]
        new_capture = metadata or {}
        new_capture[self.START_INDEX_KEY] = start_index
        # merge if capture exists
        merged = False
        for existing_capture in self._metadata[self.CAPTURE_KEY]:
            if existing_capture[self.START_INDEX_KEY] == start_index:
                existing_capture = dict_merge(existing_capture, new_capture)
                merged = True
        if not merged:
            capture_list += [new_capture]
        # sort captures by start_index
        self._metadata[self.CAPTURE_KEY] = sorted(
            capture_list,
            key=lambda item: item[self.START_INDEX_KEY],
        )

    def get_captures(self):
        """
        Returns a list of dictionaries representing all captures.
        """
        return self._metadata.get(self.CAPTURE_KEY, [])

    def get_capture_info(self, index):
        """
        Returns a dictionary containing all the capture information at sample index.
        """
        if index < self.offset:
            raise SigMFAccessError("Sample index cannot be less than dataset start offset.")
        captures = self._metadata.get(self.CAPTURE_KEY, [])
        if len(captures) == 0:
            raise SigMFAccessError("No captures in metadata.")
        cap_info = captures[0]
        for capture in captures:
            if capture[self.START_INDEX_KEY] > index:
                break
            cap_info = capture
        return cap_info

    def get_capture_start(self, index):
        """
        Returns a the start sample index of a given capture, will raise
        SigMFAccessError if this field is missing.
        """
        start = self.get_captures()[index].get(self.START_INDEX_KEY)
        if start is None:
            raise SigMFAccessError("Capture {} does not have required {} key".format(index, self.START_INDEX_KEY))
        return start

    def get_capture_byte_boundaries(self, index):
        """
        Returns a tuple of the file byte range in a dataset of a given SigMF
        capture of the form [start, stop). This function works on either
        compliant or noncompliant SigMF Recordings.
        """
        if index >= len(self.get_captures()):
            raise SigMFAccessError(
                "Invalid captures index {} (only {} captures in Recording)".format(index, len(self.get_captures()))
            )

        start_byte = 0
        prev_start_sample = 0
        for ii, capture in enumerate(self.get_captures()):
            start_byte += capture.get(self.HEADER_BYTES_KEY, 0)
            start_byte += (self.get_capture_start(ii) - prev_start_sample) * self.get_sample_size() * self.num_channels
            prev_start_sample = self.get_capture_start(ii)
            if ii >= index:
                break

        end_byte = start_byte
        if index == len(self.get_captures()) - 1:  # last captures...data is the rest of the file
            if self.data_file is not None:
                file_size = self.data_file.stat().st_size
            elif self.data_buffer is not None:
                file_size = len(self.data_buffer.getbuffer())
            else:
                raise SigMFFileError("Neither data_file nor data_buffer is available")
            end_byte = file_size - self.get_global_field(self.TRAILING_BYTES_KEY, 0)
        else:
            end_byte += (
                (self.get_capture_start(index + 1) - self.get_capture_start(index))
                * self.get_sample_size()
                * self.num_channels
            )
        return (start_byte, end_byte)

    def get_capture_byte_boundarys(self, index):
        warnings.warn(
            "get_capture_byte_boundarys() is deprecated and will be removed in a future version of sigmf. "
            "Use get_capture_byte_boundaries() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_capture_byte_boundaries(index)

    def add_annotation(self, start_index, length=None, metadata=None):
        """
        Insert annotation at start_index with length (if != None).
        """
        if start_index < self.offset:
            raise SigMFAccessError("Annotation start_index cannot be less than dataset start offset.")

        new_annot = metadata or {}
        new_annot[self.START_INDEX_KEY] = start_index
        if length is not None:
            if length <= 0:
                raise SigMFAccessError("Annotation `length` must be >= 0")
            new_annot[self.LENGTH_INDEX_KEY] = length

        self._metadata[self.ANNOTATION_KEY] += [new_annot]
        # sort annotations by start_index
        self._metadata[self.ANNOTATION_KEY] = sorted(
            self._metadata[self.ANNOTATION_KEY],
            key=lambda item: item[self.START_INDEX_KEY],
        )

    def get_annotations(self, index=None):
        """
        Get relevant annotations from metadata.

        Parameters
        ----------
        index : int, default None
            If provided returns all annotations that include this sample index.
            When omitted returns all annotations.

        Returns
        -------
        list of dict
            Each dictionary contains one annotation for the sample at `index`.
        """
        annotations = self._metadata.get(self.ANNOTATION_KEY, [])
        if index is None:
            return annotations

        annotations_including_index = []
        for annotation in annotations:
            if index < annotation[self.START_INDEX_KEY]:
                # index is before annotation starts -> skip
                continue
            if self.LENGTH_INDEX_KEY in annotation:
                # Annotation includes sample_count -> check end index
                if index >= annotation[self.START_INDEX_KEY] + annotation[self.LENGTH_INDEX_KEY]:
                    # index is after annotation end -> skip
                    continue

            annotations_including_index.append(annotation)
        return annotations_including_index

    def get_sample_size(self):
        """
        Determines the size of a sample, in bytes, from the datatype of this set.
        For complex data, a 'sample' includes both the real and imaginary part.
        """
        return dtype_info(self.datatype)["sample_size"]

    def _count_samples(self):
        """
        Count, set, and return the total number of samples in the data file.
        If there is no data file but there are annotations, use the sample_count
        from the annotation with the highest end index. If there are no annotations,
        use 0.

        For complex data, a 'sample' includes both the real and imaginary part.
        """
        if self.data_file is None and self.data_buffer is None:
            sample_count = self._get_sample_count_from_annotations()
        else:
            # if data_size_bytes is explicitly set, use it directly (already represents sample data size)
            if self.data_size_bytes is not None:
                sample_bytes = self.data_size_bytes
            else:
                # calculate from file size, subtracting header and trailing bytes
                header_bytes = sum([c.get(self.HEADER_BYTES_KEY, 0) for c in self.get_captures()])
                if self.data_file is not None:
                    file_bytes = self.data_file.stat().st_size
                elif self.data_buffer is not None:
                    file_bytes = len(self.data_buffer.getbuffer())
                else:
                    file_bytes = 0
                sample_bytes = file_bytes - self.get_global_field(self.TRAILING_BYTES_KEY, 0) - header_bytes

            total_sample_size = self.get_sample_size() * self.num_channels
            sample_count, remainder = divmod(sample_bytes, total_sample_size)
            if remainder:
                warnings.warn(
                    "Data source does not contain an integer number of samples across channels, it may be invalid."
                )
            if self._get_sample_count_from_annotations() > sample_count:
                warnings.warn("Data source ends before the final annotation in the corresponding SigMF metadata.")
        self.sample_count = sample_count
        return sample_count

    def _get_sample_count_from_annotations(self):
        """
        Returns the number of samples based on annotation with highest end index.
        NOTE: Annotations are ordered by START_INDEX_KEY and not end index, so we
        need to go through all annotations
        """
        annon_sample_count = []
        for annon in self.get_annotations():
            if self.LENGTH_INDEX_KEY in annon:
                # Annotation with sample_count
                annon_sample_count.append(annon[self.START_INDEX_KEY] + annon[self.LENGTH_INDEX_KEY])
            else:
                # Annotation without sample_count - sample count must be at least sample_start
                annon_sample_count.append(annon[self.START_INDEX_KEY])

        if annon_sample_count:
            return max(annon_sample_count)
        else:
            return 0

    def calculate_hash(self):
        """
        Calculates the hash of the data file and adds it to the global section.
        Also returns a string representation of the hash.
        """
        old_hash = self.get_global_field(self.HASH_KEY)
        if self.data_file is not None:
            new_hash = hashing.calculate_sha512(filename=self.data_file)
        else:
            new_hash = hashing.calculate_sha512(fileobj=self.data_buffer)
        if old_hash is not None:
            if old_hash != new_hash:
                raise SigMFFileError("Calculated file hash does not match associated metadata.")

        self.set_global_field(self.HASH_KEY, new_hash)
        return new_hash

    def set_data_file(
        self, data_file=None, data_buffer=None, skip_checksum=False, offset=0, size_bytes=None, map_readonly=True
    ):
        """
        Set the datafile path, then recalculate sample count.
        Update the hash and return the hash string if enabled.
        """
        if self.get_global_field(self.DATATYPE_KEY) is None:
            raise SigMFFileError("Error setting data file, the DATATYPE_KEY must be set in the global metadata first.")

        self.data_file = Path(data_file) if data_file else None
        self.data_buffer = data_buffer
        self.data_offset = offset
        self.data_size_bytes = size_bytes
        self._count_samples()

        dtype = dtype_info(self.get_global_field(self.DATATYPE_KEY))
        self.is_complex_data = dtype["is_complex"]
        num_channels = self.num_channels
        self.ndim = 1 if (num_channels < 2) else 2

        complex_int_separates = dtype["is_complex"] and dtype["is_fixedpoint"]
        mapped_dtype_size = dtype["component_size"] if complex_int_separates else dtype["sample_size"]
        mapped_length = None if size_bytes is None else size_bytes // mapped_dtype_size
        mapped_reshape = (-1,)  # we can't use -1 in mapped_length ...
        if num_channels > 1:
            mapped_reshape = mapped_reshape + (num_channels,)
        if complex_int_separates:
            # There is no corresponding numpy type, so we'll have to add another axis, with length of 2
            mapped_reshape = mapped_reshape + (2,)
        self._return_type = dtype["memmap_convert_type"]
        common_args = {"dtype": dtype["memmap_map_type"], "offset": offset}
        try:
            if self.data_file is not None:
                open_mode = "r" if map_readonly else "r+"
                memmap_shape = None if mapped_length is None else (mapped_length,)
                raveled = np.memmap(self.data_file, mode=open_mode, shape=memmap_shape, **common_args)
            elif self.data_buffer is not None:
                buffer_count = -1 if mapped_length is None else mapped_length
                raveled = np.frombuffer(self.data_buffer.getbuffer(), count=buffer_count, **common_args)
            else:
                raise SigMFFileError("In sigmffile.set_data_file(), either data_file or data_buffer must be not None")
        except SigMFFileError:  # TODO include likely exceptions here
            warnings.warn("Failed to create data array from memory-map-file or buffer!")
        else:
            self._memmap = raveled.reshape(mapped_reshape)
            self.shape = self._memmap.shape if (self._return_type is None) else self._memmap.shape[:-1]

        if self.data_file is not None:
            file_name = self.data_file.name
            ext = self.data_file.suffix
            if ext.lower() != SIGMF_DATASET_EXT:
                self.set_global_field(SigMFFile.DATASET_KEY, file_name)

        if skip_checksum:
            return None
        return self.calculate_hash()

    def validate(self):
        """
        Check schema and throw error if issue.
        """
        validate.validate(self._metadata, self.get_schema())

    def archive(self, name=None, fileobj=None):
        """Dump contents to SigMF archive format.

        `name` and `fileobj` are passed to SigMFArchive and are defined there.

        """
        archive = SigMFArchive(self, name, fileobj)
        return archive.path

    def tofile(self, file_path, pretty=True, toarchive=False, skip_validate=False):
        """
        Write metadata file or full archive containing metadata & dataset.

        Parameters
        ----------
        file_path : string
            Location to save.
        pretty : bool, default True
            When True will write more human-readable output, otherwise will be flat JSON.
        toarchive : bool, default False
            If True will write both dataset & metadata into SigMF archive format as a single `tar` file.
            If False will only write metadata to `sigmf-meta`.
        """
        if not skip_validate:
            self.validate()
        fns = get_sigmf_filenames(file_path)
        if toarchive:
            self.archive(fns["archive_fn"])
        else:
            with open(fns["meta_fn"], "w") as fp:
                self.dump(fp, pretty=pretty)
                fp.write("\n")  # text files should end in carriage return

    def read_samples_in_capture(self, index=0):
        """
        Reads samples from the specified captures segment in its entirety.

        Parameters
        ----------
        index : int, default 0
            Captures segment to read samples from.
        autoscale : bool, default True
            If dataset is in a fixed-point representation, scale samples from (min, max) to (-1.0, 1.0)

        Returns
        -------
        data : ndarray
            Samples are returned as an array of float or complex, with number of dimensions equal to NUM_CHANNELS_KEY.
        """
        cb = self.get_capture_byte_boundaries(index)
        if (cb[1] - cb[0]) % (self.get_sample_size() * self.num_channels):
            warnings.warn(
                f"Capture `{index}` in `{self.data_file}` does not contain "
                "an integer number of samples across channels. It may be invalid."
            )

        return self._read_datafile(cb[0], (cb[1] - cb[0]) // self.get_sample_size())

    def read_samples(self, start_index=0, count=-1):
        """
        Reads the specified number of samples starting at the specified index from the associated data file.

        Parameters
        ----------
        start_index : int, default 0
            Starting sample index from which to read.
        count : int, default -1
            Number of samples to read. -1 will read whole file.

        Returns
        -------
        data : ndarray
            Samples are returned as an array of float or complex, with number of dimensions equal to NUM_CHANNELS_KEY.
            Scaling behavior depends on the autoscale parameter set during construction.
        """
        if count == 0:
            raise IOError("Number of samples must be greater than zero, or -1 for all samples.")
        elif count == -1:
            count = self.sample_count - start_index
        elif start_index + count > self.sample_count:
            raise IOError("Cannot read beyond EOF.")
        if self.data_file is None and not isinstance(self.data_buffer, io.BytesIO):
            if self.get_global_field(self.METADATA_ONLY_KEY, False):
                # only if data_file is `None` allows access to dynamically generated datsets
                raise SigMFFileError("Cannot read samples from a metadata only distribution.")
            else:
                raise SigMFFileError("No signal data file has been associated with the metadata.")
        first_byte = start_index * self.get_sample_size() * self.num_channels
        return self._read_datafile(first_byte, count * self.num_channels)

    def _read_datafile(self, first_byte, nitems):
        """
        internal function for reading samples from datafile
        """
        dtype = dtype_info(self.get_global_field(self.DATATYPE_KEY))
        self.is_complex_data = dtype["is_complex"]
        is_fixedpoint_data = dtype["is_fixedpoint"]
        is_unsigned_data = dtype["is_unsigned"]
        data_type_in = dtype["sample_dtype"]
        component_size = dtype["component_size"]

        data_type_out = np.dtype("f4") if not self.is_complex_data else np.dtype("f4, f4")
        num_channels = self.num_channels

        if self.data_file is not None:
            fp = open(self.data_file, "rb")
            # account for data_offset when seeking (important for NCDs)
            seek_position = first_byte + getattr(self, "data_offset", 0)
            fp.seek(seek_position, 0)

            data = np.fromfile(fp, dtype=data_type_in, count=nitems)
        elif self.data_buffer is not None:
            # handle offset for data_buffer like we do for data_file
            buffer_data = self.data_buffer.getbuffer()[first_byte:]
            data = np.frombuffer(buffer_data, dtype=data_type_in, count=nitems)
        else:
            data = self._memmap

        if num_channels != 1:
            # return reshaped view for num_channels
            # first dimension will be double size if `is_complex_data`
            data = data.reshape(data.shape[0] // num_channels, num_channels)
        data = data.astype(data_type_out)
        if self.autoscale and is_fixedpoint_data:
            data = data.view(np.dtype("f4"))
            if is_unsigned_data:
                data -= 2 ** (component_size * 8 - 1)
            data *= 2 ** -(component_size * 8 - 1)
            data = data.view(data_type_out)
        if self.is_complex_data:
            data = data.view(np.complex64)

        if self.data_file is not None:
            fp.close()

        return data


class SigMFCollection(SigMFMetafile):
    VERSION_KEY = "core:version"
    DESCRIPTION_KEY = "core:description"
    AUTHOR_KEY = "core:author"
    COLLECTION_DOI_KEY = "core:collection_doi"
    LICENSE_KEY = "core:license"
    EXTENSIONS_KEY = "core:extensions"
    STREAMS_KEY = "core:streams"
    COLLECTION_KEY = "collection"
    VALID_COLLECTION_KEYS = [
        AUTHOR_KEY,
        COLLECTION_DOI_KEY,
        DESCRIPTION_KEY,
        EXTENSIONS_KEY,
        LICENSE_KEY,
        STREAMS_KEY,
        VERSION_KEY,
    ]
    VALID_KEYS = {COLLECTION_KEY: VALID_COLLECTION_KEYS}

    def __init__(
        self, metafiles: list = None, metadata: dict = None, base_path=None, skip_checksums: bool = False
    ) -> None:
        """
        Create a SigMF Collection object.

        Parameters
        ----------
        metafiles: list, optional
            A list of SigMF metadata filenames objects comprising the Collection.
            There should be at least one file.
        metadata: dict, optional
            Collection metadata to use, if not provided this will populate a minimal set of default metadata.
            The `core:streams` field will be regenerated automatically.
        base_path : str | bytes | PathLike, optional
            Base path of the collection recordings.
        skip_checksums : bool, optional
            If true will skip calculating checksum on datasets.

        Raises
        ------
        SigMFError
            If metadata files do not exist.
        """
        super().__init__()
        self.skip_checksums = skip_checksums

        if base_path is None:
            self.base_path = Path("")
        else:
            self.base_path = Path(base_path)

        if metadata is None:
            self._metadata = {self.COLLECTION_KEY: {}}
            self._metadata[self.COLLECTION_KEY][self.STREAMS_KEY] = []
        else:
            self._metadata = metadata

        if metafiles is None:
            self.metafiles = []
        else:
            self.set_streams(metafiles)

        # set version to current implementation
        self.set_collection_field(self.VERSION_KEY, __specification__)

        if not self.skip_checksums:
            self.verify_stream_hashes()

    def __len__(self) -> int:
        """
        The length of a collection is the number of streams.
        """
        return len(self.get_stream_names())

    def verify_stream_hashes(self) -> None:
        """
        Compares the stream hashes in the collection metadata to the metadata files.

        Raises
        ------
        SigMFFileError
            If any dataset checksums do not match saved metadata.
        """
        streams = self.get_collection_field(self.STREAMS_KEY, [])
        for stream in streams:
            old_hash = stream.get("hash")
            metafile_name = get_sigmf_filenames(stream.get("name"))["meta_fn"]
            metafile_path = self.base_path / metafile_name
            if Path.is_file(metafile_path):
                new_hash = hashing.calculate_sha512(filename=metafile_path)
                if old_hash != new_hash:
                    raise SigMFFileError(
                        f"Calculated file hash for {metafile_path} does not match collection metadata."
                    )

    def set_streams(self, metafiles) -> None:
        """
        Configures the collection `core:streams` field from the specified list of metafiles.
        """
        self.metafiles = metafiles
        streams = []
        for metafile in self.metafiles:
            metafile_path = self.base_path / metafile
            if metafile.endswith(".sigmf-meta") and Path.is_file(metafile_path):
                stream = {
                    # name must be string here to be serializable later
                    "name": str(get_sigmf_filenames(metafile)["base_fn"]),
                    "hash": hashing.calculate_sha512(filename=metafile_path),
                }
                streams.append(stream)
            else:
                raise SigMFFileError(f"Specifed stream file {metafile_path} is not a valid SigMF Metadata file")
        self.set_collection_field(self.STREAMS_KEY, streams)

    def get_stream_names(self) -> list:
        """
        Returns a list of `name` object(s) from the `collection` level `core:streams` metadata.
        """
        return [s.get("name") for s in self.get_collection_field(self.STREAMS_KEY, [])]

    def set_collection_info(self, new_collection: dict) -> None:
        """
        Overwrite the collection info with a new dictionary.
        """
        self._metadata[self.COLLECTION_KEY] = new_collection.copy()

    def get_collection_info(self) -> dict:
        """
        Returns a dictionary with all the collection info.
        """
        try:
            return self._metadata.get(self.COLLECTION_KEY, {})
        except AttributeError:
            return {}

    def set_collection_field(self, key: str, value) -> None:
        """
        Inserts a value into the collection field.
        """
        self._metadata[self.COLLECTION_KEY][key] = value

    def get_collection_field(self, key: str, default=None):
        """
        Return a field from the collection info, or default if the field is not set.
        """
        return self._metadata[self.COLLECTION_KEY].get(key, default)

    def tofile(self, file_path, pretty: bool = True) -> None:
        """
        Write metadata file

        Parameters
        ----------
        file_path : string
            Location to save.
        pretty : bool, default True
            When True will write more human-readable output, otherwise will be flat JSON.
        """
        filenames = get_sigmf_filenames(file_path)
        with open(filenames["collection_fn"], "w") as handle:
            self.dump(handle, pretty=pretty)
            handle.write("\n")  # text files should end in carriage return

    def get_SigMFFile(self, stream_name=None, stream_index=None):
        """
        Returns the SigMFFile instance of the specified stream if it exists
        """
        if stream_name is not None and stream_name not in self.get_stream_names():
            # invalid stream name
            return
        if stream_index is not None and stream_index < len(self):
            stream_name = self.get_stream_names()[stream_index]
        if stream_name is not None:
            metafile = get_sigmf_filenames(stream_name)["meta_fn"]
            metafile_path = self.base_path / metafile
            return fromfile(metafile_path, skip_checksum=self.skip_checksums)


def dtype_info(datatype):
    """
    Parses a datatype string conforming to the SigMF spec and returns a dict
    of values describing the format.

    Keyword arguments:
    datatype -- a SigMF-compliant datatype string
    """
    if datatype is None:
        raise SigMFFileError("Invalid datatype 'None'.")
    output_info = {}
    dtype = datatype.lower()

    is_unsigned_data = "u" in datatype
    is_complex_data = "c" in datatype
    is_fixedpoint_data = "f" not in datatype

    dtype = datatype.lower().split("_")

    byte_order = ""
    if len(dtype) == 2:
        if dtype[1][0] == "l":
            byte_order = "<"
        elif dtype[1][0] == "b":
            byte_order = ">"
        else:
            raise SigMFFileError("Unrecognized endianness specifier: '{}'".format(dtype[1]))
    dtype = dtype[0]
    if "64" in dtype:
        sample_size = 8
    elif "32" in dtype:
        sample_size = 4
    elif "16" in dtype:
        sample_size = 2
    elif "8" in dtype:
        sample_size = 1
    else:
        raise SigMFFileError("Unrecognized datatype: '{}'".format(dtype))
    component_size = sample_size
    if is_complex_data:
        sample_size *= 2
    sample_size = int(sample_size)

    data_type_str = byte_order
    data_type_str += "f" if not is_fixedpoint_data else "u" if is_unsigned_data else "i"
    data_type_str += str(component_size)

    memmap_convert_type = None
    if is_complex_data:
        data_type_str = ",".join((data_type_str, data_type_str))
        memmap_map_type = byte_order
        if is_fixedpoint_data:
            memmap_map_type += ("u" if is_unsigned_data else "i") + str(component_size)
            memmap_convert_type = byte_order + "c8"
        else:
            memmap_map_type += "c" + str(sample_size)
    else:
        memmap_map_type = data_type_str

    data_type_in = np.dtype(data_type_str)
    output_info["sample_dtype"] = data_type_in
    output_info["component_dtype"] = data_type_in["f0"] if is_complex_data else data_type_in
    output_info["sample_size"] = sample_size
    output_info["component_size"] = component_size
    output_info["is_complex"] = is_complex_data
    output_info["is_unsigned"] = is_unsigned_data
    output_info["is_fixedpoint"] = is_fixedpoint_data
    output_info["memmap_map_type"] = memmap_map_type
    output_info["memmap_convert_type"] = memmap_convert_type
    return output_info


def get_dataset_filename_from_metadata(meta_fn, metadata=None):
    """
    Parse provided metadata and return the expected data filename.

    In the case of a metadata-only distribution, or if the file does not exist,
    this will return ``None``.

    Priority for conflicting datasets:

    1. Use the file named ``<stem>.SIGMF_DATASET_EXT`` if it exists.
    2. Use the file in the ``DATASET_KEY`` field (non-compliant dataset) if it exists.
    3. Return ``None`` (may be a metadata-only distribution).
    """
    compliant_filename = get_sigmf_filenames(meta_fn)["data_fn"]
    noncompliant_filename = metadata["global"].get(SigMFFile.DATASET_KEY, None)

    if Path.is_file(compliant_filename):
        if noncompliant_filename:
            warnings.warn(
                f"Compliant Dataset `{compliant_filename}` exists but "
                f"{SigMFFile.DATASET_KEY} is also defined; using `{compliant_filename}`"
            )
        return compliant_filename

    elif noncompliant_filename:
        dir_path = Path(meta_fn).parent
        noncompliant_data_file_path = Path.joinpath(dir_path, noncompliant_filename)
        if Path.is_file(noncompliant_data_file_path):
            if metadata["global"].get(SigMFFile.METADATA_ONLY_KEY, False):
                raise SigMFFileError(
                    f"Schema defines {SigMFFile.DATASET_KEY} "
                    f"but {SigMFFile.METADATA_ONLY_KEY} also exists; using `{noncompliant_filename}`"
                )
            return noncompliant_data_file_path
        else:
            raise SigMFFileError(
                f"Non-Compliant Dataset `{noncompliant_filename}` is specified in {SigMFFile.DATASET_KEY} "
                "but does not exist!"
            )
    return None


def fromarchive(archive_path, dir=None, skip_checksum=False, autoscale=True):
    """Extract an archive and return a SigMFFile.

    The `dir` parameter is no longer used as this function has been changed to
    access SigMF archives without extracting them.

    Parameters
    ----------
    archive_path: str
        Path to `sigmf-archive` tarball.
    dir: str, optional
        No longer used. Kept for compatibility.
    skip_checksum: bool, default False
        Skip dataset checksum calculation.
    autoscale: bool, default True
        If dataset is in a fixed-point representation, scale samples from (min, max) to (-1.0, 1.0).

    Returns
    -------
    SigMFFile
        Instance created from archive.
    """
    from .archivereader import SigMFArchiveReader

    return SigMFArchiveReader(archive_path, skip_checksum=skip_checksum, autoscale=autoscale).sigmffile


def fromfile(filename, skip_checksum=False, autoscale=True):
    """
    Read a file as a SigMFFile or SigMFCollection.

    The file can be one of:
    * a SigMF Archive (.sigmf)
    * a SigMF Metadata file (.sigmf-meta)
    * a SigMF Dataset file (.sigmf-data)
    * a SigMF Collection file (.sigmf-collection)
    * a non-SigMF RF recording that can be converted (.wav, .cdif)

    Parameters
    ----------
    filename: str | bytes | PathLike
        Path for SigMF Metadata, Dataset, Archive or Collection (with or without extension).
    skip_checksum: bool, default False
        When True will not read entire dataset to calculate hash.
    autoscale: bool, default True
        If dataset is in a fixed-point representation, scale samples from (min, max) to (-1.0, 1.0).

    Returns
    -------
    SigMFFile | SigMFCollection
        A SigMFFile or a SigMFCollection depending on file type.

    Raises
    ------
    SigMFFileError
        If the file cannot be read as any supported format.
    SigMFConversionError
        If auto-detection conversion fails.
    """
    file_path = Path(filename)
    fns = get_sigmf_filenames(filename)
    meta_fn = fns["meta_fn"]
    archive_fn = fns["archive_fn"]
    collection_fn = fns["collection_fn"]

    # extract the extension to check file type
    ext = file_path.suffix.lower()

    # group SigMF extensions for cleaner checking
    sigmf_extensions = (SIGMF_METADATA_EXT, SIGMF_DATASET_EXT, SIGMF_COLLECTION_EXT, SIGMF_ARCHIVE_EXT)

    # try SigMF archive
    if (ext.endswith(SIGMF_ARCHIVE_EXT) or not Path.is_file(meta_fn)) and Path.is_file(archive_fn):
        return fromarchive(archive_fn, skip_checksum=skip_checksum, autoscale=autoscale)

    # try SigMF collection
    if (ext.endswith(SIGMF_COLLECTION_EXT) or not Path.is_file(meta_fn)) and Path.is_file(collection_fn):
        collection_fp = open(collection_fn, "rb")
        bytestream_reader = codecs.getreader("utf-8")
        mdfile_reader = bytestream_reader(collection_fp)
        metadata = json.load(mdfile_reader)
        collection_fp.close()

        dir_path = meta_fn.parent
        return SigMFCollection(metadata=metadata, base_path=dir_path, skip_checksums=skip_checksum)

    # try standard SigMF metadata file
    if Path.is_file(meta_fn):
        meta_fp = open(meta_fn, "rb")
        bytestream_reader = codecs.getreader("utf-8")
        mdfile_reader = bytestream_reader(meta_fp)
        metadata = json.load(mdfile_reader)
        meta_fp.close()

        data_fn = get_dataset_filename_from_metadata(meta_fn, metadata)
        return SigMFFile(metadata=metadata, data_file=data_fn, skip_checksum=skip_checksum, autoscale=autoscale)

    # try auto-detection for non-SigMF files only
    if Path.is_file(file_path) and not ext.endswith(sigmf_extensions):
        if not autoscale:
            # TODO: allow autoscale=False for converters
            warnings.warn("non-SigMF auto-detection conversion only supports autoscale=True; ignoring autoscale=False")
        magic_bytes = get_magic_bytes(file_path, count=4, offset=0)

        if magic_bytes == b"RIFF":
            from .convert.wav import wav_to_sigmf

            return wav_to_sigmf(file_path, create_ncd=True)

        elif magic_bytes == b"BLUE":
            from .convert.blue import blue_to_sigmf

            return blue_to_sigmf(file_path, create_ncd=True)

    # if file doesn't exist at all or no valid files found, raise original error
    raise SigMFFileError(f"Cannot read {filename} as SigMF or supported non-SigMF format.")


def get_sigmf_filenames(filename):
    """
    Safely returns a set of SigMF file paths given an input filename.

    Parameters
    ----------
    filename : str | bytes | PathLike
        The SigMF filename with any extension.

    Returns
    -------
    dict with filename keys.
    """
    stem_path = Path(filename)
    # If the path has a sigmf suffix, remove it. Otherwise do not remove the
    # suffix, because the filename might contain '.' characters which are part
    # of the filename rather than an extension.
    sigmf_suffixes = [
        SIGMF_DATASET_EXT,
        SIGMF_METADATA_EXT,
        SIGMF_ARCHIVE_EXT,
        SIGMF_COLLECTION_EXT,
    ]
    if stem_path.suffix in sigmf_suffixes:
        with_suffix_path = stem_path
        stem_path = stem_path.with_suffix("")
    else:
        # Add a dummy suffix to prevent the .with_suffix() calls below from
        # overriding part of the filename which is interpreted as a suffix
        with_suffix_path = stem_path.with_name(f"{stem_path.name}{SIGMF_DATASET_EXT}")

    return {
        "base_fn": stem_path,
        "data_fn": with_suffix_path.with_suffix(SIGMF_DATASET_EXT),
        "meta_fn": with_suffix_path.with_suffix(SIGMF_METADATA_EXT),
        "archive_fn": with_suffix_path.with_suffix(SIGMF_ARCHIVE_EXT),
        "collection_fn": with_suffix_path.with_suffix(SIGMF_COLLECTION_EXT),
    }

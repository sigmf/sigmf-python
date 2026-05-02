# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""SigMF metadata key and file extension constants."""

# ---------------------------------------------------------------------------
# capture / annotation sample indexing keys
# ---------------------------------------------------------------------------
SAMPLE_START_KEY = "core:sample_start"
SAMPLE_COUNT_KEY = "core:sample_count"
GLOBAL_INDEX_KEY = "core:global_index"
OFFSET_KEY = "core:offset"

# ---------------------------------------------------------------------------
# data specification keys
# ---------------------------------------------------------------------------
NUM_CHANNELS_KEY = "core:num_channels"
SHA512_KEY = "core:sha512"
VERSION_KEY = "core:version"
DATATYPE_KEY = "core:datatype"
FREQUENCY_KEY = "core:frequency"
HEADER_BYTES_KEY = "core:header_bytes"
FREQ_LOWER_EDGE_KEY = "core:freq_lower_edge"
FREQ_UPPER_EDGE_KEY = "core:freq_upper_edge"
SAMPLE_RATE_KEY = "core:sample_rate"
TRAILING_BYTES_KEY = "core:trailing_bytes"

# ---------------------------------------------------------------------------
# metadata / descriptive keys
# ---------------------------------------------------------------------------
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
METADATA_ONLY_KEY = "core:metadata_only"
EXTENSIONS_KEY = "core:extensions"
DATETIME_KEY = "core:datetime"
UUID_KEY = "core:uuid"

# ---------------------------------------------------------------------------
# location keys
# ---------------------------------------------------------------------------
LATITUDE_KEY = "core:latitude"
LONGITUDE_KEY = "core:longitude"
GEOLOCATION_KEY = "core:geolocation"

# ---------------------------------------------------------------------------
# cross-reference keys
# ---------------------------------------------------------------------------
COLLECTION_KEY = "core:collection"

# ---------------------------------------------------------------------------
# collection-specific field keys
# ---------------------------------------------------------------------------
COLLECTION_DOI_KEY = "core:collection_doi"
STREAMS_KEY = "core:streams"

# ---------------------------------------------------------------------------
# valid key lists per section
# ---------------------------------------------------------------------------
VALID_GLOBAL_KEYS = [
    AUTHOR_KEY,
    COLLECTION_KEY,
    DATASET_KEY,
    DATATYPE_KEY,
    DATA_DOI_KEY,
    DESCRIPTION_KEY,
    EXTENSIONS_KEY,
    GEOLOCATION_KEY,
    SHA512_KEY,
    HW_KEY,
    LICENSE_KEY,
    META_DOI_KEY,
    METADATA_ONLY_KEY,
    NUM_CHANNELS_KEY,
    RECORDER_KEY,
    SAMPLE_RATE_KEY,
    OFFSET_KEY,
    TRAILING_BYTES_KEY,
    VERSION_KEY,
]

VALID_CAPTURE_KEYS = [
    DATETIME_KEY,
    FREQUENCY_KEY,
    GLOBAL_INDEX_KEY,
    HEADER_BYTES_KEY,
    SAMPLE_START_KEY,
]

VALID_ANNOTATION_KEYS = [
    COMMENT_KEY,
    FREQ_UPPER_EDGE_KEY,
    FREQ_LOWER_EDGE_KEY,
    GENERATOR_KEY,
    LABEL_KEY,
    LATITUDE_KEY,
    LONGITUDE_KEY,
    SAMPLE_COUNT_KEY,
    SAMPLE_START_KEY,
    UUID_KEY,
]

VALID_COLLECTION_KEYS = [
    AUTHOR_KEY,
    COLLECTION_DOI_KEY,
    DESCRIPTION_KEY,
    EXTENSIONS_KEY,
    LICENSE_KEY,
    STREAMS_KEY,
    VERSION_KEY,
]

# ---------------------------------------------------------------------------
# file extension constants
# ---------------------------------------------------------------------------
SIGMF_ARCHIVE_EXT = ".sigmf"
SIGMF_METADATA_EXT = ".sigmf-meta"
SIGMF_DATASET_EXT = ".sigmf-data"
SIGMF_COLLECTION_EXT = ".sigmf-collection"

SIGMF_COMPRESSED_EXTS = {
    "gz": ".sigmf.gz",
    "xz": ".sigmf.xz",
    "zip": ".sigmf.zip",
}

# all recognized archive extensions (uncompressed + compressed)
SIGMF_ARCHIVE_EXTS = {SIGMF_ARCHIVE_EXT} | set(SIGMF_COMPRESSED_EXTS.values())

# all SigMF file suffixes
SIGMF_SUFFIXES = [
    SIGMF_DATASET_EXT,
    SIGMF_METADATA_EXT,
    SIGMF_ARCHIVE_EXT,
    SIGMF_COLLECTION_EXT,
]

# ---------------------------------------------------------------------------
# deprecated alias map — used by _SigMFDeprecatingMeta in sigmffile.py
# maps old_name -> (new_name, value)
# ---------------------------------------------------------------------------
_DEPRECATED_ALIASES = {
    "START_INDEX_KEY": ("SAMPLE_START_KEY", SAMPLE_START_KEY),
    "LENGTH_INDEX_KEY": ("SAMPLE_COUNT_KEY", SAMPLE_COUNT_KEY),
    "START_OFFSET_KEY": ("OFFSET_KEY", OFFSET_KEY),
    "HASH_KEY": ("SHA512_KEY", SHA512_KEY),
    "FLO_KEY": ("FREQ_LOWER_EDGE_KEY", FREQ_LOWER_EDGE_KEY),
    "FHI_KEY": ("FREQ_UPPER_EDGE_KEY", FREQ_UPPER_EDGE_KEY),
    "LAT_KEY": ("LATITUDE_KEY", LATITUDE_KEY),
    "LON_KEY": ("LONGITUDE_KEY", LONGITUDE_KEY),
}

# ---------------------------------------------------------------------------
# public exports
# ---------------------------------------------------------------------------
__all__ = [
    # sample indexing
    "SAMPLE_START_KEY",
    "SAMPLE_COUNT_KEY",
    "GLOBAL_INDEX_KEY",
    "OFFSET_KEY",
    # data specification
    "NUM_CHANNELS_KEY",
    "SHA512_KEY",
    "VERSION_KEY",
    "DATATYPE_KEY",
    "FREQUENCY_KEY",
    "HEADER_BYTES_KEY",
    "FREQ_LOWER_EDGE_KEY",
    "FREQ_UPPER_EDGE_KEY",
    "SAMPLE_RATE_KEY",
    "TRAILING_BYTES_KEY",
    # metadata / descriptive
    "COMMENT_KEY",
    "DESCRIPTION_KEY",
    "AUTHOR_KEY",
    "META_DOI_KEY",
    "DATA_DOI_KEY",
    "GENERATOR_KEY",
    "LABEL_KEY",
    "RECORDER_KEY",
    "LICENSE_KEY",
    "HW_KEY",
    "DATASET_KEY",
    "METADATA_ONLY_KEY",
    "EXTENSIONS_KEY",
    "DATETIME_KEY",
    "UUID_KEY",
    # location
    "LATITUDE_KEY",
    "LONGITUDE_KEY",
    "GEOLOCATION_KEY",
    # cross-reference
    "COLLECTION_KEY",
    # collection-specific
    "COLLECTION_DOI_KEY",
    "STREAMS_KEY",
    # valid key lists
    "VALID_GLOBAL_KEYS",
    "VALID_CAPTURE_KEYS",
    "VALID_ANNOTATION_KEYS",
    "VALID_COLLECTION_KEYS",
    # file extensions
    "SIGMF_ARCHIVE_EXT",
    "SIGMF_METADATA_EXT",
    "SIGMF_DATASET_EXT",
    "SIGMF_COLLECTION_EXT",
    "SIGMF_COMPRESSED_EXTS",
    "SIGMF_ARCHIVE_EXTS",
    "SIGMF_SUFFIXES",
]

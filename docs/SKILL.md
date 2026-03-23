# SigMF Python Skill

You have access to the `sigmf` Python library for working with Signal Metadata Format (SigMF) recordings — a standard for describing datasets of digital signals.

## When To Use

Use `sigmf` when the user wants to:
- Read, write, or inspect RF/signal recording metadata
- Convert between signal formats (WAV, BLUE/Platinum, SigMF archives)
- Create or validate SigMF-compliant datasets
- Query signal data: sample rate, frequency, timestamps, annotations
- Build signal processing pipelines that need structured metadata

## Core Concepts

- **SigMFFile**: Metadata + data for a single recording
- **SigMFCollection**: Groups multiple SigMFFile recordings
- **SigMFArchive** (`.sigmf`): Tarball containing `.sigmf-meta` + `.sigmf-data`
- **SigMFArchiveReader**: Read archives without extracting

## Quick Reference

### Reading Recordings

```python
import sigmf

# Read SigMF recording
recording = sigmf.fromfile("recording.sigmf-meta")

# Read SigMF archive
recording = sigmf.fromfile("recording.sigmf")

# Read WAV/BLUE/other formats as SigMF
recording = sigmf.fromfile("recording.wav")
recording = sigmf.fromfile("recording.cdif")

# Read archive without extracting
reader = sigmf.SigMFArchiveReader("recording.sigmf")
recording = reader.get_SigMFFile()
```

### Accessing Samples

```python
# Get first 1024 samples (returns numpy array)
samples = recording[0:1024]

# Iteration
for sample in recording:
    process(sample)

# Sample count
count = len(recording)

# Sample rate
rate = recording.sample_rate
```

### Metadata Access

```python
# Get/set global fields
rate = recording.get_global_field("core:sample_rate")
recording.set_global_field("core:sample_rate", 20e6)
recording.sample_rate = 20e6  # shorthand for core: fields

# All core globals accessible as attributes
freq = recording.frequency          # core:frequency
dtype = recording.datatype          # core:datatype
author = recording.author           # core:author
desc = recording.description        # core:description
hw = recording.recorder             # core:recorder
sha = recording.sha512              # core:sha512

# Get all global info
global_info = recording.get_global_info()

# Set multiple global fields at once
recording.set_global_info({"core:sample_rate": 20e6, "core:frequency": 2.4e9})
```

### Captures

```python
# Add capture (defines when a segment of recording begins)
recording.add_capture(0, metadata={
    "core:frequency": 2.4e9,
    "core:datetime": "2026-01-15T10:30:00Z"
})

# Get all captures
captures = recording.get_captures()

# Get capture info by index
info = recording.get_capture_info(0)
```

### Annotations

```python
# Add annotation (marks a region of interest in the signal)
recording.add_annotation(
    start_index=1000,
    length=512,
    metadata={
        "core:label": "BLE Beacon",
        "core:comment": "Detected BLE advertisement",
        "core:freq_lower_edge": 2.402e9,
        "core:freq_upper_edge": 2.480e9
    }
)

# Get annotations (optionally filtered by index)
annotations = recording.get_annotations()
annotations_in_range = recording.get_annotations(index=500)
```

### Writing & Exporting

```python
# Write metadata JSON to file
recording.tofile("output.sigmf-meta")

# Write as SigMF archive
recording.tofile("output.sigmf", toarchive=True)

# Archive to buffer
import io
buf = io.BytesIO()
archive = recording.archive("my_recording", fileobj=buf)
```

### Collections

```python
from sigmf import SigMFCollection

collection = SigMFCollection()
collection.add_sigmf_file("recording1.sigmf-meta")
collection.add_sigmf_file("recording2.sigmf-meta")
collection.set_collection_field("core:author", "SigMF")
collection.tofile("collection.sigmf-collection")
```

### Validation

```python
# Validate metadata against spec
errors = recording.validate()
if errors:
    print(f"Validation issues: {errors}")

# Calculate and verify SHA-512 checksum
recording.calculate_hash()
```

### Utility Functions

```python
from sigmf.utils import (
    get_sigmf_iso8601_datetime_now,  # Current time as SigMF datetime string
    parse_iso8601_datetime,           # Parse SigMF datetime string
    get_endian_str,                   # Get endian string from numpy array
    get_data_type_str,                # Get SigMF datatype string from numpy array
)
```

## Common Patterns

### Convert WAV to SigMF

```python
import sigmf
recording = sigmf.fromfile("input.wav")
recording.set_global_field("core:frequency", 2.4e9)
recording.tofile("output.sigmf", toarchive=True)
```

### Create SigMF from numpy data

```python
import numpy as np
import sigmf

samples = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex64)
meta = sigmf.SigMFFile(
    data_file="samples.sigmf-data",
    global_info={
        "core:datatype": "cf32_le",
        "core:sample_rate": 1e6,
        "core:num_channels": 1,
    }
)
meta.add_capture(0)
meta.tofile("output.sigmf-meta")
# Also write the raw data file
samples.tofile("samples.sigmf-data")
```

### Annotate a recording

```python
recording = sigmf.fromfile("recording.sigmf-meta")
recording.add_annotation(
    start_index=4096,
    length=2048,
    metadata={
        "core:label": "WiFi Beacon",
        "core:freq_lower_edge": 2.412e9,
        "core:freq_upper_edge": 2.412e9,
        "core:comment": "802.11 beacon frame detected"
    }
)
recording.tofile("output.sigmf-meta")
```

## Tips

- SigMF metadata files are JSON — always valid JSON with `global`, `captures`, `annotations` top-level keys
- Dataset files are raw binary — the `core:datatype` field specifies format (e.g., `cf32_le`, `rf32_le`, `ci16_le`)
- Use `autoscale=True` (default) to automatically scale fixed-point samples to [-1.0, 1.0]
- Archives (`.sigmf`) are tar files — use `SigMFArchiveReader` for read-only access without extraction
- `fromfile()` auto-detects format by magic bytes — works with WAV, BLUE/Platinum, SigMF, and others

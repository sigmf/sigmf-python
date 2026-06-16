![Rendered SigMF Logo](https://raw.githubusercontent.com/sigmf/SigMF/refs/heads/main/logo/sigmf_logo.png)

[![PyPI Version Shield](https://img.shields.io/pypi/v/sigmf)](https://pypi.org/project/SigMF/)
[![Build Status Shield](https://img.shields.io/github/actions/workflow/status/sigmf/sigmf-python/main.yml)](https://github.com/sigmf/sigmf-python/actions?query=branch%3Amain)
[![License Shield](https://img.shields.io/pypi/l/sigmf)](https://en.wikipedia.org/wiki/GNU_Lesser_General_Public_License)
[![Documentation Shield](https://img.shields.io/readthedocs/sigmf)](https://sigmf.readthedocs.io/en/latest/)
[![PyPI Downloads Shield](https://img.shields.io/pypi/dm/sigmf)](https://pypi.org/project/SigMF/)

The `sigmf` library makes it easy to interact with Signal Metadata Format
(SigMF) recordings. This library is compatible with Python 3.7-3.14 and is distributed
freely under the terms GNU Lesser GPL v3 License.

This module follows the SigMF specification [html](https://sigmf.org/)/[pdf](https://sigmf.github.io/SigMF/sigmf-spec.pdf) from the [spec repository](https://github.com/sigmf/SigMF).

### Install

```bash
pip install sigmf
# or
conda install sigmf
# or
mamba install sigmf
```

### Read SigMF

```python
import sigmf

# read SigMF recording
meta = sigmf.fromfile("recording.sigmf-meta")
samples = meta[0:1024]  # get first 1024 samples
sample_rate = meta.sample_rate  # get sample rate

# read compressed SigMF archives
meta = sigmf.fromfile("recording.sigmf.gz")   # gzip-compressed
meta = sigmf.fromfile("recording.sigmf.xz")   # xz-compressed
meta = sigmf.fromfile("recording.sigmf.zip")  # zip archive

# read other formats containing RF time series as SigMF
meta = sigmf.fromfile("recording.wav")   # WAV
meta = sigmf.fromfile("recording.cdif")  # BLUE / Platinum
meta = sigmf.fromfile("recording.xml")   # Signal Hound Spike
```

### Write SigMF

```python
import numpy as np
import sigmf

data = np.array([0.1 + 0.2j, 0.3 + 0.4j], dtype=np.complex64)
meta = sigmf.fromarray(data)
# optional additional metadata
meta.sample_rate = 8000
meta.description = "sample recording"
meta.add_capture(start_index=0, metadata={sigmf.FREQUENCY_KEY: 915e6})
# creates recording.sigmf-data and recording.sigmf-meta
meta.tofile("recording")
# or create compressed archive
meta.tofile("recording.sigmf.gz")
```

### HDF5 metadata sidecar (optional)

For recordings with very large `captures`/`annotations`, the optional
`hdf5-meta` extension can write a columnar HDF5 sidecar next to the
`.sigmf-meta` file. The JSON metadata stays complete and authoritative; the
sidecar is a smaller, faster cache for column-oriented access. Requires the
optional `h5py` dependency: `pip install sigmf[hdf5]`.

```python
import sigmf
from sigmf import hdf5

# write the sidecar alongside the JSON (declares the hdf5-meta extension)
meta.tofile("recording", write_hdf5=True)  # also writes recording.sigmf-meta.h5

# fast columnar read: open ONLY the sidecar, no JSON parsing, no per-row dicts
with hdf5.open("recording.sigmf-meta.h5") as fast:
    starts = fast.annotations_column("core:sample_start")  # numpy column
    table = fast.annotations_array()                       # structured array

# or discover via the JSON once, then prefer the sidecar when present & fresh
fast = hdf5.fromfile("recording.sigmf-meta")  # SigMFFileHDF5 if usable, else SigMFFile

# the standard reader is unchanged and always reads pure JSON
meta = sigmf.fromfile("recording.sigmf-meta")
```

### Docs

**[Please visit our documentation for full API reference and more info.](https://sigmf.readthedocs.io/en/latest/)**

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
```

### Read SigMF

```python
import sigmf

# read SigMF recording
meta = sigmf.fromfile("recording.sigmf-meta")
samples = meta[0:1024]  # get first 1024 samples
sample_rate = meta.sample_rate  # get sample rate


# read other formats containing RF time series as SigMF
meta = sigmf.fromfile("recording.wav")   # WAV
meta = sigmf.fromfile("recording.cdif")  # BLUE / Platinum
```

### Docs

**[Please visit our documentation for full API reference and more info.](https://sigmf.readthedocs.io/en/latest/)**

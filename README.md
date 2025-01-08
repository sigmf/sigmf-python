<p align="center"><img src="https://github.com/sigmf/SigMF/blob/v1.2.0/logo/sigmf_logo.png" alt="Rendered SigMF Logo"/></p>

This python module makes it easy to interact with Signal Metadata Format
(SigMF) recordings. This module works with Python 3.7+ and is distributed
freely under the terms GNU Lesser GPL v3 License.

This module follows the SigMF specification [html](https://sigmf.org/)/[pdf](https://sigmf.github.io/SigMF/sigmf-spec.pdf) from the [spec repository](https://github.com/sigmf/SigMF).

# Installation

To install the latest PyPi release, install from pip:

```bash
pip install sigmf
```

To install the latest git release, build from source:

```bash
git clone https://github.com/sigmf/sigmf-python.git
cd sigmf-python
pip install .
```

Testing can be run with a variety of tools:

```bash
# pytest and coverage run locally
pytest
coverage run
# run coverage in a venv
tox run
# other useful tools
pylint sigmf tests
pytype
black
flake8
```

# Examples

### Load a SigMF archive; read all samples & metadata

```python
import sigmf
handle = sigmf.sigmffile.fromfile('example.sigmf')
handle.read_samples() # returns all timeseries data
handle.get_global_info() # returns 'global' dictionary
handle.get_captures() # returns list of 'captures' dictionaries
handle.get_annotations() # returns list of all annotations
```

### Verify SigMF dataset integrity & compliance

```bash
sigmf_validate example.sigmf
```

### Load a SigMF dataset; read its annotation, metadata, and samples

```python
from sigmf import SigMFFile, sigmffile

# Load a dataset
filename = 'logo/sigmf_logo' # extension is optional
signal = sigmffile.fromfile(filename)

# Get some metadata and all annotations
sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
sample_count = signal.sample_count
signal_duration = sample_count / sample_rate
annotations = signal.get_annotations()

# Iterate over annotations
for adx, annotation in enumerate(annotations):
    annotation_start_idx = annotation[SigMFFile.START_INDEX_KEY]
    annotation_length = annotation[SigMFFile.LENGTH_INDEX_KEY]
    annotation_comment = annotation.get(SigMFFile.COMMENT_KEY, "[annotation {}]".format(adx))

    # Get capture info associated with the start of annotation
    capture = signal.get_capture_info(annotation_start_idx)
    freq_center = capture.get(SigMFFile.FREQUENCY_KEY, 0)
    freq_min = freq_center - 0.5*sample_rate
    freq_max = freq_center + 0.5*sample_rate

    # Get frequency edges of annotation (default to edges of capture)
    freq_start = annotation.get(SigMFFile.FLO_KEY)
    freq_stop = annotation.get(SigMFFile.FHI_KEY)

    # Get the samples corresponding to annotation
    samples = signal.read_samples(annotation_start_idx, annotation_length)
```

### Create and save a Collection of SigMF Recordings from numpy arrays

First, create a single SigMF Recording and save it to disk

```python
import datetime as dt
import numpy as np
import sigmf
from sigmf import SigMFFile
from sigmf.utils import get_data_type_str, get_sigmf_iso8601_datetime_now

# suppose we have an complex timeseries signal
data = np.zeros(1024, dtype=np.complex64)

# write those samples to file in cf32_le
data.tofile('example_cf32.sigmf-data')

# create the metadata
meta = SigMFFile(
    data_file='example_cf32.sigmf-data', # extension is optional
    global_info = {
        SigMFFile.DATATYPE_KEY: get_data_type_str(data),  # in this case, 'cf32_le'
        SigMFFile.SAMPLE_RATE_KEY: 48000,
        SigMFFile.AUTHOR_KEY: 'jane.doe@domain.org',
        SigMFFile.DESCRIPTION_KEY: 'All zero complex float32 example file.',
    }
)

# create a capture key at time index 0
meta.add_capture(0, metadata={
    SigMFFile.FREQUENCY_KEY: 915000000,
    SigMFFile.DATETIME_KEY: get_sigmf_iso8601_datetime_now(),
})

# add an annotation at sample 100 with length 200 & 10 KHz width
meta.add_annotation(100, 200, metadata = {
    SigMFFile.FLO_KEY: 914995000.0,
    SigMFFile.FHI_KEY: 915005000.0,
    SigMFFile.COMMENT_KEY: 'example annotation',
})

# check for mistakes & write to disk
meta.tofile('example_cf32.sigmf-meta') # extension is optional
```

Now lets add another SigMF Recording and associate them with a SigMF Collection:

```python
from sigmf import SigMFCollection

data_ci16 = np.zeros(1024, dtype=np.complex64)

#rescale and save as a complex int16 file:
data_ci16 *= pow(2, 15)
data_ci16.view(np.float32).astype(np.int16).tofile('example_ci16.sigmf-data')

# create the metadata for the second file
meta_ci16 = SigMFFile(
    data_file='example_ci16.sigmf-data', # extension is optional
    global_info = {
        SigMFFile.DATATYPE_KEY: 'ci16_le', # get_data_type_str() is only valid for numpy types
        SigMFFile.SAMPLE_RATE_KEY: 48000,
        SigMFFile.DESCRIPTION_KEY: 'All zero complex int16 file.',
    }
)
meta_ci16.add_capture(0, metadata=meta.get_capture_info(0))
meta_ci16.tofile('example_ci16.sigmf-meta')

collection = SigMFCollection(['example_cf32.sigmf-meta', 'example_ci16.sigmf-meta'],
        metadata = {'collection': {
            SigMFCollection.AUTHOR_KEY: 'sigmf@sigmf.org',
            SigMFCollection.DESCRIPTION_KEY: 'Collection of two all zero files.',
        }
    }
)
streams = collection.get_stream_names()
sigmf = [collection.get_SigMFFile(stream) for stream in streams]
collection.tofile('example_zeros.sigmf-collection')
```

The SigMF Collection and its associated Recordings can now be loaded like this:

```python
from sigmf import sigmffile
collection = sigmffile.fromfile('example_zeros')
ci16_sigmffile = collection.get_SigMFFile(stream_name='example_ci16')
cf32_sigmffile = collection.get_SigMFFile(stream_name='example_cf32')
```

### Load a SigMF Archive and slice its data without untaring it

Since an *archive* is merely a tarball (uncompressed), and since there any many
excellent tools for manipulating tar files, it's fairly straightforward to
access the *data* part of a SigMF archive without un-taring it. This is a
compelling feature because __1__ archives make it harder for the `-data` and
the `-meta` to get separated, and __2__ some datasets are so large that it can
be impractical (due to available disk space, or slow network speeds if the
archive file resides on a network file share) or simply obnoxious to untar it
first.

```python
>>> import sigmf
>>> arc = sigmf.SigMFArchiveReader('/src/LTE.sigmf')
>>> arc.shape
(15379532,)
>>> arc.ndim
1
>>> arc[:10]
array([-20.+11.j, -21. -6.j, -17.-20.j, -13.-52.j,   0.-75.j,  22.-58.j,
        48.-44.j,  49.-60.j,  31.-56.j,  23.-47.j], dtype=complex64)
```

The preceeding example exhibits another feature of this approach; the archive
`LTE.sigmf` is actually `complex-int16`'s on disk, for which there is no
corresponding type in `numpy`. However, the `.sigmffile` member keeps track of
this, and converts the data to `numpy.complex64` *after* slicing it, that is,
after reading it from disk.

```python
>>> arc.sigmffile.get_global_field(sigmf.SigMFFile.DATATYPE_KEY)
'ci16_le'

>>> arc.sigmffile._memmap.dtype
dtype('int16')

>>> arc.sigmffile._return_type
'<c8'
```

Another supported mode is the case where you might have an archive that *is not
on disk* but instead is simply `bytes` in a python variable.
Instead of needing to write this out to a temporary file before being able to
read it, this can be done "in mid air" or "without touching the ground (disk)".

```python
>>> import sigmf, io
>>> sigmf_bytes = io.BytesIO(open('/src/LTE.sigmf', 'rb').read())
>>> arc = sigmf.SigMFArchiveReader(archive_buffer=sigmf_bytes)
>>> arc[:10]
array([-20.+11.j, -21. -6.j, -17.-20.j, -13.-52.j,   0.-75.j,  22.-58.j,
        48.-44.j,  49.-60.j,  31.-56.j,  23.-47.j], dtype=complex64)
```

# Frequently Asked Questions

### Is this a GNU Radio effort?

*No*, this is not a GNU Radio-specific effort.
This effort first emerged from a group of GNU Radio core
developers, but the goal of the project to provide a standard that will be
useful to anyone and everyone, regardless of tool or workflow.

### Is this specific to wireless communications?

*No*, similar to the response, above, the goal is to create something that is
generally applicable to _signal processing_, regardless of whether or not the
application is communications related.

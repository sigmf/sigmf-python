==========
Quickstart
==========

Here we discuss how to do all basic operations with SigMF.

.. _install:

-------
Install
-------

To install the latest PyPi release, install from pip:

.. code-block:: console

    $ pip install sigmf

----------------------
Read a SigMF Recording
----------------------

.. code-block:: python

    import sigmf
    handle = sigmf.fromfile("example.sigmf")
    # reading data
    handle.read_samples() # read all timeseries data
    handle[10:50] # read memory slice of samples 10 through 50
    # accessing metadata
    handle.sample_rate # get sample rate attribute
    handle.get_global_info() # returns 'global' dictionary
    handle.get_captures() # returns list of 'captures' dictionaries
    handle.get_annotations() # returns list of all annotations

-----------------------------------
Verify SigMF Integrity & Compliance
-----------------------------------

.. code-block:: console

    $ sigmf_validate example.sigmf

---------------------------------------
Save a Numpy array as a SigMF Recording
---------------------------------------

.. code-block:: python

    import numpy as np
    from sigmf import SigMFFile
    from sigmf.utils import get_data_type_str, get_sigmf_iso8601_datetime_now

    # suppose we have a complex timeseries signal
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

----------------------------------
Attribute Access for Global Fields
----------------------------------

SigMF-Python provides convenient attribute-style access for core global metadata fields,
allowing you to read and write metadata using simple dot notation alongside the traditional
method-based approach.

.. code-block:: python

    import numpy as np
    from sigmf import SigMFFile

    # create a new recording
    meta = SigMFFile()

    # set global metadata using attributes
    meta.sample_rate = 48000
    meta.author = 'jane.doe@domain.org'
    meta.datatype = 'cf32_le'
    meta.description = 'Example recording with attribute access'
    meta.license = 'MIT'
    meta.recorder = 'hackrf_one'

    # read global metadata using attributes
    print(f"Sample rate: {meta.sample_rate}")
    print(f"Author: {meta.author}")
    print(f"License: {meta.license}")

    # method-based approach
    meta.set_global_field(SigMFFile.HW_KEY, 'SDR Hardware v1.2')
    hw_info = meta.get_global_field(SigMFFile.HW_KEY)
    print(f"Hardware: {hw_info}")

    # attribute and method access are equivalent
    meta.set_global_field(SigMFFile.RECORDER_KEY, 'usrp_b210')
    print(f"Recorder via attribute: {meta.recorder}")  # prints: usrp_b210

.. note::

   Only core **global** fields support attribute access. Capture and annotation
   fields must still be accessed using the traditional ``add_capture()`` and
   ``add_annotation()`` methods.

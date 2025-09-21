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
    handle.read_samples() # returns all timeseries data
    handle.get_global_info() # returns 'global' dictionary
    handle.get_captures() # returns list of 'captures' dictionaries
    handle.get_annotations() # returns list of all annotations
    handle[10:50] # return memory slice of samples 10 through 50

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

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
    data.tofile("example.sigmf-data")

    # create the metadata
    meta = SigMFFile(
        data_file="example.sigmf-data", # extension is optional
        global_info = {
            SigMFFile.DATATYPE_KEY: get_data_type_str(data),  # in this case, "cf32_le"
            SigMFFile.SAMPLE_RATE_KEY: 48000,
            SigMFFile.AUTHOR_KEY: "jane.doe@domain.org",
            SigMFFile.DESCRIPTION_KEY: "All zero complex float32 example file.",
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
        SigMFFile.COMMENT_KEY: "example annotation",
    })

    # validate & write to disk
    meta.tofile("example.sigmf-meta") # extension is optional

----------------------------------
Attribute Access for Global Fields
----------------------------------

SigMF-Python provides convenient attribute read/write access for core global
metadata fields, allowing you use simple dot notation alongside the traditional
method-based approach.

.. code-block:: python

    import sigmf

    # read some recording
    meta = sigmf.SigMFFile("sigmf_logo")

    # read global metadata
    print(f"Sample rate: {meta.sample_rate}")
    print(f"License: {meta.license}")

    # set global metadata
    meta.description = "Updated description via attribute access"
    meta.author = "jane.doe@domain.org"

    # validate & write changes to disk
    meta.tofile("sigmf_logo_updated")

.. note::

   Only core **global** fields support attribute access. Capture and annotation
   fields must still be accessed using the traditional ``get_captures()`` and
   ``get_annotations()`` methods.

--------------------------------
Control Fixed-Point Data Scaling
--------------------------------

For fixed-point datasets, you can control whether samples are automatically scaled to floating-point values:

.. code-block:: python

    import sigmf

    # Default behavior: autoscale fixed-point data to [-1.0, 1.0] range
    handle = sigmf.fromfile("fixed_point_data.sigmf")
    samples = handle.read_samples()  # Returns float32/complex64

    # Disable autoscaling to access raw integer values
    handle_raw = sigmf.fromfile("fixed_point_data.sigmf", autoscale=False)
    raw_samples = handle_raw.read_samples()  # Returns original integer types

    # Both slicing and read_samples() respect the autoscale setting
    assert handle[0:10].dtype == handle.read_samples(count=10).dtype

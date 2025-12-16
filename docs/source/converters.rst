=================
Format Converters
=================

The SigMF Python library includes converters to import data from various file formats into SigMF format.
These converters make it easy to migrate existing RF recordings to the standardized SigMF format while preserving metadata when possible.

Overview
--------

Converters are available for:

* **BLUE files** - MIDAS Blue and Platinum BLUE RF recordings (``.cdif``)
* **WAV files** - Audio recordings (``.wav``)

All converters return a :class:`~sigmf.SigMFFile` object that can be used immediately or saved to disk.
Converters preserve datatypes and metadata where possible.


Command Line Usage
~~~~~~~~~~~~~~~~~~

Converters can be used from the command line after ``pip install sigmf``:

.. code-block:: bash

    sigmf_convert_blue input.cdif
    sigmf_convert_wav input.wav

or by using module syntax:

.. code-block:: bash

    python3 -m sigmf.convert.blue input.cdif
    python3 -m sigmf.convert.wav input.wav


BLUE Converter
--------------

The BLUE converter handles CDIF (.cdif) recordings while placing BLUE header information into the following global fields:

* ``blue:fixed`` - fixed header information (at start of file)
* ``blue:adjunct`` - adjunct header information (after fixed header)
* ``blue:extended`` - extended header information (at end of file)
* ``blue:keywords`` - user-defined key-value pairs

.. autofunction:: sigmf.convert.blue.blue_to_sigmf


.. code-block:: python

    from sigmf.convert.blue import blue_to_sigmf

    # read BLUE, write SigMF, and return SigMFFile object
    meta = blue_to_sigmf(blue_path="recording.cdif", out_path="recording.sigmf")

    # access converted data
    samples = meta.read_samples()
    sample_rate_hz = meta.sample_rate

    # access BLUE-specific metadata
    blue_type = meta.get_global_field("blue:fixed")["type"] # e.g., 1000
    blue_version = meta.get_global_field("blue:keywords")["IO"] # e.g., "X-Midas"


WAV Converter
-------------

This is useful when working with audio datasets.

.. autofunction:: sigmf.convert.wav.wav_to_sigmf


.. code-block:: python

    from sigmf.convert.wav import wav_to_sigmf

    # read WAV, write SigMF, and return SigMFFile object
    meta = wav_to_sigmf(wav_path="recording.wav", out_path="recording.sigmf")

    # access converted data
    samples = meta.read_samples()
    sample_rate_hz = meta.sample_rate
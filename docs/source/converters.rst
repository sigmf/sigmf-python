=================
Format Converters
=================

The SigMF Python library includes converters to import data from various file formats into SigMF format.
Converters can create standard SigMF file pairs or Non-Conforming Datasets (NCDs) that reference the original files.

Overview
--------

Converters are available for:

* **BLUE files** - MIDAS Blue and Platinum BLUE RF recordings (``.cdif``)
* **WAV files** - Audio recordings (``.wav``)

All converters return a :class:`~sigmf.SigMFFile` object. Auto-detection is available through :func:`~sigmf.sigmffile.fromfile`.


Auto-Detection
~~~~~~~~~~~~~~

The :func:`~sigmf.sigmffile.fromfile` function automatically detects file formats and creates Non-Conforming Datasets:

.. code-block:: python

    import sigmf

    # auto-detect and create NCD for any supported format
    meta = sigmf.fromfile("recording.cdif")  # BLUE file
    meta = sigmf.fromfile("recording.wav")   # WAV file
    meta = sigmf.fromfile("recording.sigmf")  # SigMF archive

    samples = meta.read_samples()


Command Line Usage
~~~~~~~~~~~~~~~~~~

Converters can be used from the command line:

.. code-block:: bash

    sigmf_convert_blue recording.cdif
    sigmf_convert_wav recording.wav

or by using module execution:

.. code-block:: bash

    python -m sigmf.convert.blue recording.cdif
    python -m sigmf.convert.wav recording.wav


Output Options
~~~~~~~~~~~~~~

Converters support multiple output modes:

* **Standard conversion**: Creates ``.sigmf-data`` and ``.sigmf-meta`` files
* **Archive mode**: Creates single ``.sigmf`` archive with ``--archive``
* **Non-Conforming Dataset**: Creates metadata-only file referencing original data with ``--ncd``


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

    # standard conversion
    meta = blue_to_sigmf(blue_path="recording.cdif", out_path="recording")

    # create NCD automatically (metadata-only, references original file)
    meta = blue_to_sigmf(blue_path="recording.cdif")

    # access standard SigMF data & metadata
    all_samples = meta.read_samples()
    sample_rate_hz = meta.sample_rate

    # access BLUE-specific metadata
    blue_type = meta.get_global_field("blue:fixed")["type"]  # e.g., 1000
    blue_version = meta.get_global_field("blue:keywords")["IO"]  # e.g., "X-Midas"


WAV Converter
-------------

Converts WAV audio recordings to SigMF format.

.. autofunction:: sigmf.convert.wav.wav_to_sigmf

.. code-block:: python

    from sigmf.convert.wav import wav_to_sigmf

    # standard conversion
    meta = wav_to_sigmf(wav_path="recording.wav", out_path="recording")

    # create NCD automatically (metadata-only, references original file)
    meta = wav_to_sigmf(wav_path="recording.wav")

    # access standard SigMF data & metadata
    all_samples = meta.read_samples()
    sample_rate_hz = meta.sample_rate
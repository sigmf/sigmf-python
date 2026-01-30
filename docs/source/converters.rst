==========
Converters
==========

The SigMF Python library includes converters to import data from various RF recording formats into SigMF.
Converters can create standard SigMF file pairs or Non-Conforming Datasets (NCDs) that reference the original files.

Overview
--------

Conversion is available for:

* **BLUE files** - MIDAS Blue and Platinum BLUE RF recordings (usually ``.cdif``)
* **WAV files** - Audio recordings (``.wav``)

All converters return a :class:`~sigmf.SigMFFile` object with converted metadata.


Fromfile Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~

The :func:`~sigmf.sigmffile.fromfile` function automatically detects input file
formats and reads without writing any output files:

.. code-block:: python

    import sigmf

    # auto-detect and create NCD for any supported format
    meta = sigmf.fromfile("recording.cdif")  # BLUE file
    meta = sigmf.fromfile("recording.wav")   # WAV file
    meta = sigmf.fromfile("recording.sigmf")  # SigMF archive

    all_samples = meta.read_samples()
    sample_rate = meta.sample_rate


Python API
~~~~~~~~~~~

For programmatic access, use the individual converter functions directly:

.. code-block:: python

    from sigmf.convert.wav import wav_to_sigmf
    from sigmf.convert.blue import blue_to_sigmf

    # convert WAV to SigMF archive
    _ = wav_to_sigmf(wav_path="recording.wav", out_path="recording", create_archive=True)

    # convert BLUE to SigMF pair and return metadata for new files
    meta = blue_to_sigmf(blue_path="recording.cdif", out_path="recording")


Command Line Usage
~~~~~~~~~~~~~~~~~~

Converters are accessed through a unified command-line interface that automatically detects file formats:

.. code-block:: bash

    # unified converter
    sigmf_convert input_file output_file

    # examples
    sigmf_convert recording.cdif recording.sigmf
    sigmf_convert recording.wav recording.sigmf

The converter uses magic byte detection to automatically identify BLUE and WAV file formats.
No need to remember format-specific commands!


Output Options
~~~~~~~~~~~~~~

The unified converter supports multiple output modes:

.. code-block:: bash

    # standard conversion (creates out.sigmf-data and out.sigmf-meta files)
    sigmf_convert in.wav out

    # archive mode (creates single out.sigmf archive)
    sigmf_convert in.wav out --archive

    # non-conforming dataset (creates out.sigmf-meta only, references original file)
    sigmf_convert in.wav out --ncd

    # extra verbose output
    sigmf_convert in.wav out -vv

**Important**: When using ``--ncd``, the input and output files must be in the same directory.
This ensures proper relative path references in the metadata.


BLUE Converter
--------------

The BLUE converter handles CDIF (.cdif) recordings while placing BLUE header information into the following global fields:

* ``blue:fixed`` - Fixed header information (at start of file).
* ``blue:adjunct`` - Adjunct header information (after fixed header).
* ``blue:extended`` - Extended header information (at end of file). Note any duplicate fields will have a suffix like ``_1``, ``_2``, etc appended.

.. autofunction:: sigmf.convert.blue.blue_to_sigmf

Examples
~~~~~~~~

.. code-block:: python

    from sigmf.convert.blue import blue_to_sigmf

    # standard conversion
    meta = blue_to_sigmf(blue_path="recording.cdif", out_path="recording")

    # create NCD automatically (metadata-only, references original file) but don't save any output file
    meta = blue_to_sigmf(blue_path="recording.cdif")

    # access standard SigMF data & metadata
    all_samples = meta.read_samples()
    sample_rate = meta.sample_rate

    # access BLUE-specific metadata
    blue_type = meta.get_global_field("blue:fixed")["type"]  # e.g., 1000
    blue_version = meta.get_global_field("blue:fixed")["keywords"]["IO"]  # e.g., "X-Midas"

Tested Formats
~~~~~~~~~~~~~~

BLUE files use a 2-digit format code where the first is the sample type (row) and the second is the sample size (column).
For example ``SB`` contains real values with 8 bits per sample and ``CF`` contains complex values with 32 bits per component (64 bits per sample).

The following table summarizes tested BLUE formats and their compatibility with the converter:

.. csv-table::
    :header-rows: 1
    :stub-columns: 1

    "Code", ":abbr:`P (packed)`", ":abbr:`N (int4)`", ":abbr:`B (int8)`", ":abbr:`U (uint16)`", ":abbr:`I (int16)`", ":abbr:`V (uint32)`", ":abbr:`L (int32)`", ":abbr:`F (float32)`", ":abbr:`X (int64)`", ":abbr:`D (float64)`", ":abbr:`O (excess-128)`"
    ":abbr:`S (scalar)`", "❌", "❌", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "❌"
    ":abbr:`C (complex)`", "❌", "❌", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "✅", "❌"

**Legend:**
    * ✅ = Tested and known working
    * ❌ = Unsupported


WAV Converter
-------------

Converts WAV audio recordings to SigMF format.

.. autofunction:: sigmf.convert.wav.wav_to_sigmf

Examples
~~~~~~~~

.. code-block:: python

    from sigmf.convert.wav import wav_to_sigmf

    # standard conversion
    meta = wav_to_sigmf(wav_path="recording.wav", out_path="recording")

    # create NCD automatically (metadata-only, references original file)
    meta = wav_to_sigmf(wav_path="recording.wav")

    # access standard SigMF data & metadata
    all_samples = meta.read_samples()
    sample_rate_hz = meta.sample_rate
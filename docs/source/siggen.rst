================
Signal Generator
================

The :class:`~sigmf.siggen.SigMFGenerator` class provides a builder-pattern API
for creating synthetic RF signals as SigMF files. It is intended for generating
test signals and example files during development and testing.

Signals are complex baseband (``cf32_le``) and support tones, linear frequency
sweeps, additive white Gaussian noise, and frequency/phase offsets. All
parameters can be left unspecified and randomized with a seed for
reproducibility.

-----------
Basic Usage
-----------

.. code-block:: python

    from sigmf.siggen import SigMFGenerator

    # generate a 1 kHz tone at 48 kHz sample rate for 1 second
    signal = SigMFGenerator().tone(1000).sample_rate(48000).duration(1.0).generate()
    signal.read_samples()  # complex64 numpy array

The returned object is a standard :class:`~sigmf.sigmffile.SigMFFile` backed by
an in-memory buffer, so all the usual metadata and data access methods work.

--------------------
Combining Components
--------------------

Chain multiple ``.tone()`` and ``.sweep()`` calls to build multi-component
signals. Each component gets its own annotation with frequency bounds and a
label.

.. code-block:: python

    signal = (
        SigMFGenerator()
        .tone(1000)
        .tone(-2500)
        .sweep(500, 4000)
        .sample_rate(48000)
        .duration(0.5)
        .snr(20)
        .generate()
    )

    # each component is annotated individually
    for annotation in signal.get_annotations():
        print(annotation)

---------------------
Random Test Signals
---------------------

Calling ``.generate()`` with no components produces a fully random signal.
A seed ensures reproducibility across runs.

.. code-block:: python

    # deterministic random signal
    signal = SigMFGenerator(seed=0xDEADBEEF).generate()

    # the number and type of components are randomly chosen
    print(signal.description)  # e.g. "synthetic signal with 3 tones and 2 sweeps"
    print(signal.get_annotations())  # one annotation per component

Without a seed, each call produces a different signal.

----------------------------
Metadata & Annotations
----------------------------

Annotations are automatically created for each signal component, noise floor,
and any applied frequency or phase offsets. Metadata fields like description,
author, and comment can be set via the builder.

.. code-block:: python

    signal = (
        SigMFGenerator()
        .tone(440)
        .sample_rate(44100)
        .duration(1.0)
        .snr(25)
        .frequency_offset(1000)
        .author("test@example.com")
        .description("test tone with noise")
        .comment("for unit testing")
        .generate()
    )

    print(signal.get_global_info())
    print(signal.get_annotations())

---------
API
---------

.. autoclass:: sigmf.siggen.SigMFGenerator
   :members:
   :undoc-members:

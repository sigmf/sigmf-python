Usage
=====

.. _installation:

Installation
------------

To install the latest PyPi release, install from pip:

.. code-block:: console

   $ pip install sigmf

To install the latest git release, build from source:

.. code-block:: console

   $ git clone https://github.com/sigmf/sigmf-python.git
   $ cd sigmf-python
   $ pip install .

Testing
-------

Testing can be run locally:

.. code-block:: console

   $ coverage run

Run coverage on multiple python versions:

.. code-block:: console

   $ tox run

Other tools developers may want to use:

.. code-block:: console

   $ pytest -rA tests/test_archive.py # test one file verbosely
   $ pylint sigmf tests # lint entire project
   $ black sigmf tests # autoformat entire project
   $ isort sigmf tests # format imports for entire project

Examples
--------

Load a SigMF archive; read all samples & metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import sigmf
   handle = sigmf.sigmffile.fromfile("example.sigmf")
   handle.read_samples() # returns all timeseries data
   handle.get_global_info() # returns 'global' dictionary
   handle.get_captures() # returns list of 'captures' dictionaries
   handle.get_annotations() # returns list of all annotations

Verify SigMF dataset integrity & compliance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

   $ sigmf_validate example.sigmf

TODO: Insert more examples from `README.md`.

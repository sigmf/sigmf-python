==========
Developers
==========

This page is for developers of the ``sigmf-python`` module.

-------
Install
-------

To install from source:

.. code-block:: console

   $ git clone https://github.com/sigmf/sigmf-python.git
   $ cd sigmf-python
   $ pip install .[test]

-------
Testing
-------

This library contains many tests in the ``tests/`` folder. These can all be run locally:

.. code-block:: console

   $ coverage run

Or tests can be run within a temporary environment on all supported python versions:

.. code-block:: console

   $ tox run

To run a single (perhaps new) test that may be needed verbosely:

.. code-block:: console

   $ pytest -rA tests/test_archive.py

To lint the entire project and get suggested changes:

.. code-block:: console

   $ ruff check

To autoformat the entire project according to our coding standard:

.. code-block:: console

   $ ruff format

----
Docs
----

To build the docs and host locally:

.. code-block:: console

   $ cd docs
   $ pip install -r requirements.txt
   $ make clean
   $ make html
   $ python3 -m http.server --directory build/html/

--------------
Find an Issue?
--------------

Issues can be addressed by opening an `issue
<https://github.com/sigmf/sigmf-python/issues>`_ or by forking the project and
submitting a `pull request <https://github.com/sigmf/sigmf-python/pulls>`_.

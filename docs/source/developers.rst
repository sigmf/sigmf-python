==========
Developers
==========

This page is for developers of the ``sigmf-python`` module.

-------
Install
-------

To install the latest git release, build from source:

.. code-block:: console

   $ git clone https://github.com/sigmf/sigmf-python.git
   $ cd sigmf-python
   $ pip install .

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

   $ pylint sigmf tests

To autoformat the entire project according to our coding standard:

.. code-block:: console

   $ black sigmf tests # autoformat entire project
   $ isort sigmf tests # format imports for entire project

----
Docs
----

To build the docs and host locally:

.. code-block:: console

   $ cd docs
   $ make clean
   $ make html
   $ python3 -m http.server --directory build/html/

--------------
Find an Issue?
--------------

Issues can be addressed by opening an `issue
<https://github.com/sigmf/sigmf-python/issues>`_ or by forking the project and
submitting a `pull request <https://github.com/sigmf/sigmf-python/pulls>`_.

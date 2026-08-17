==========================
Frequently Asked Questions
==========================

.. contents::
    :local:

..
    Frequently asked questions should be questions that actually got asked.
    Formulate them as a question and an answer.
    Consider that the answer is best as a reference to another place in the documentation.

----------------------------------------
Can I use my own custom metadata fields?
----------------------------------------

*Yes*, you can add arbitrary fields to the ``global``, ``captures``, and
``annotations`` objects. However, we recommend defining custom fields in a
SigMF extension and listing that extension in ``core:extensions`` so that other
tools can understand your metadata:

.. code-block:: json

    "global": {
        "core:extensions": [
            {
                "name": "my-extension",
                "version": "0.0.1",
                "optional": true
            }
        ],
        "my-extension:my_field": "some value"
    }

If you think your extension will be useful to others, consider publishing it or
submitting it to the `SigMF Community Extensions repository
<https://github.com/sigmf/community-extensions>`_.

---------------------------
Is this a GNU Radio effort?
---------------------------

*No*, this is not a GNU Radio-specific effort.
This effort first emerged from a group of GNU Radio core
developers, but the goal of the project to provide a standard that will be
useful to anyone and everyone, regardless of tool or workflow.

--------------------------------------------
Is this specific to wireless communications?
--------------------------------------------

*No*, similar to the response, above, the goal is to create something that is
generally applicable to *signal processing*, regardless of whether or not the
application is communications related.

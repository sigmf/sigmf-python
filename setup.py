#!/usr/bin/env python3
from setuptools import setup
import os
import re

short_description = "Python module for interacting with SigMF recordings."

with open("README.md", encoding="utf-8") as handle:
    long_description = handle.read()

with open(os.path.join("sigmf", "__init__.py"), encoding="utf-8") as handle:
    version = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', handle.read()).group(1)

setup(
    name="SigMF",
    version=version,
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sigmf/sigmf-python",
    license="GNU Lesser General Public License v3 or later (LGPLv3+)",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "sigmf_validate = sigmf.validate:main",
            "sigmf_gui = sigmf.gui:main [gui]",
        ]
    },
    packages=["sigmf"],
    package_data={
        "sigmf": ["*.json"],
    },
    install_requires=["numpy", "jsonschema"],
    extras_require={"gui": "pysimplegui==4.0.0"},
    setup_requires=["pytest-runner"],
    tests_require=["pytest>3", "hypothesis"],
    zip_safe=False,
)

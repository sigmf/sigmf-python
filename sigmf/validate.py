# Copyright: Multiple Authors
#
# This file is part of sigmf-python. https://github.com/sigmf/sigmf-python
#
# SPDX-License-Identifier: LGPL-3.0-or-later

"""SigMF Validator"""
import argparse
import glob
import json
import logging
import os
import sys

# multi-threading library - should work well as I/O will be the primary
# cost for small SigMF files. Swap to ProcessPool if files are large.
from concurrent.futures import ThreadPoolExecutor, as_completed

# required for Python 3.7
from typing import Optional, Tuple

import jsonschema

from . import __version__ as toolversion
from . import error, schema, sigmffile


def validate(metadata, ref_schema=schema.get_schema()) -> None:
    """
    Check that the provided `metadata` dict is valid according to the `ref_schema` dict.
    Walk entire schema and check all keys.

    Parameters
    ----------
    metadata : dict
        The SigMF metadata to be validated.
    ref_schema : dict, optional
        The schema that holds the SigMF metadata definition.
        Since the schema evolves over time, we may want to be able to check
        against different versions in the *future*.

    Raises
    ------
    ValidationError
        If metadata is invalid.
    """
    jsonschema.validators.validate(instance=metadata, schema=ref_schema)

    # ensure captures and annotations have monotonically increasing sample_start
    for key in ["captures", "annotations"]:
        count = -1
        for item in metadata[key]:
            new_count = item["core:sample_start"]
            if new_count < count:
                raise jsonschema.exceptions.ValidationError(f"{key} has incorrect sample start ordering.")
            count = new_count


def _validate_single_file(filename, skip_checksum: bool, logger: logging.Logger) -> int:
    """Validates a single SigMF file.

    To be called as part of a multithreading / multiprocess application.

    Parameters
    ----------
    filename : str
        Path and name to sigmf.data or sigmf.meta file.
    skip_checksum : bool
        Whether to perform checksum computation.
    logger : logging.Logger
        Logging object to log errors to.

    Returns
    -------
    rc : int
        0 if OK, 1 if err
    """
    try:
        # load signal
        signal = sigmffile.fromfile(filename, skip_checksum=skip_checksum)
        # validate
        signal.validate()

    # handle any of 4 exceptions at once...
    except (jsonschema.exceptions.ValidationError, error.SigMFFileError, json.decoder.JSONDecodeError, IOError) as err:
        # catch the error, log, and continue
        logger.error(f"file `{filename}`: {err}")
        return 1
    else:
        return 0


def main(arg_tuple: Optional[Tuple[str, ...]] = None) -> None:
    """entry-point for command-line validator"""
    parser = argparse.ArgumentParser(
        description="Validate SigMF Archive or file pair against JSON schema.", prog="sigmf_validate"
    )
    parser.add_argument("path", nargs="*", help="SigMF path(s). Accepts * wildcards and extensions are optional.")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip reading dataset to validate checksum.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--version", action="version", version=f"%(prog)s {toolversion}")

    # allow pass-in arg_tuple for testing purposes
    args = parser.parse_args(arg_tuple)

    level_lut = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    log = logging.getLogger()
    logging.basicConfig(level=level_lut[min(args.verbose, 2)])

    paths = []
    # resolve possible wildcards
    for path in args.path:
        paths += glob.glob(path)

    # multi-processing / threading pathway.
    n_completed = 0
    n_total = len(paths)
    # estimate number of CPU cores
    # https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-using-python
    est_cpu_cores = len(os.sched_getaffinity(0))
    # create a thread pool
    # https://docs.python.org/3.7/library/concurrent.futures.html#threadpoolexecutor
    with ThreadPoolExecutor(max_workers=est_cpu_cores - 1) as executor:
        # submit jobs
        future_validations = {executor.submit(_validate_single_file, path, args.skip_checksum, log) for path in paths}
        # load and await jobs to complete... no return
        for future in as_completed(future_validations):
            if future.result() == 0:
                n_completed += 1

    if n_total == 0:
        log.error("No paths to validate.")
        sys.exit(1)
    elif n_completed != n_total:
        log.info(f"Validated {n_completed} of {n_total} files OK")
        sys.exit(1)
    else:
        log.info(f"Validated all {n_total} files OK!")


if __name__ == "__main__":
    main()

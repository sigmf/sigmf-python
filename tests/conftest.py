# Copyright 2017 GNU Radio Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tempfile

import pytest

from sigmf.sigmffile import SigMFFile

from .testdata import (TEST_FLOAT32_DATA_1,
                       TEST_METADATA_1,
                       TEST_FLOAT32_DATA_2,
                       TEST_METADATA_2,
                       TEST_FLOAT32_DATA_3,
                       TEST_METADATA_3)


@pytest.fixture
def test_data_file_1():
    with tempfile.NamedTemporaryFile() as temp:
        TEST_FLOAT32_DATA_1.tofile(temp.name)
        yield temp


@pytest.fixture
def test_data_file_2():
    with tempfile.NamedTemporaryFile() as t:
        TEST_FLOAT32_DATA_2.tofile(t.name)
        yield t


@pytest.fixture
def test_data_file_3():
    with tempfile.NamedTemporaryFile() as t:
        TEST_FLOAT32_DATA_3.tofile(t.name)
        yield t


@pytest.fixture
def test_sigmffile(test_data_file_1):
    f = SigMFFile(name='test1')
    f.set_global_field("core:datatype", "rf32_le")
    f.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA_1))
    f.add_capture(start_index=0)
    f.set_data_file(test_data_file_1.name)
    assert f._metadata == TEST_METADATA_1
    return f


@pytest.fixture
def test_alternate_sigmffile(test_data_file_2):
    f = SigMFFile(name='test2')
    f.set_global_field("core:datatype", "rf32_le")
    f.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA_2))
    f.add_capture(start_index=0)
    f.set_data_file(test_data_file_2.name)
    assert f._metadata == TEST_METADATA_2
    return f


@pytest.fixture
def test_alternate_sigmffile_2(test_data_file_3):
    f = SigMFFile(name='test3')
    f.set_global_field("core:datatype", "rf32_le")
    f.add_annotation(start_index=0, length=len(TEST_FLOAT32_DATA_3))
    f.add_capture(start_index=0)
    f.set_data_file(test_data_file_3.name)
    assert f._metadata == TEST_METADATA_3
    return f

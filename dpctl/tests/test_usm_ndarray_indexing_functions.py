#                      Data Parallel Control (dpctl)
#
# Copyright 2020-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.testing import assert_array_equal

import pytest
from helper import get_queue_or_skip, skip_if_dtype_not_supported

import dpctl.tensor as dpt


def test_take_arg_validation():
    q = get_queue_or_skip()

    shape = (3, 3)
    Xnp = np.arange(np.prod(shape)).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    Ynp = np.arange(2, dtype="int")
    Y = dpt.asarray(Ynp, sycl_queue=q)

    with pytest.raises(TypeError):
        dpt.take(Xnp, Y, axis=1)
    with pytest.raises(TypeError):
        dpt.take(X, Ynp, axis=1)

    Y = dpt.astype(Y, "float")
    with pytest.raises(TypeError):
        dpt.take(X, Y, axis=1)


@pytest.mark.parametrize(
    "dtype",
    [
        "b1",
        "i1",
        "u1",
        "i2",
        "u2",
        "i4",
        "u4",
        "i8",
        "u8",
        "f2",
        "f4",
        "f8",
        "c8",
        "c16",
        ]
)
def test_take_dtypes(dtype):
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dtype, q)

    shape = (1, 2, 3)
    X = dpt.reshape(dpt.arange(np.prod(shape), sycl_queue=q), shape)
    indices = dpt.asarray([0], dtype="int", sycl_queue=q)
    Y = dpt.take(X, indices, axis=0)

    assert X.dtype == Y.dtype


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((4, 3), 0),
        ((5, 5, 5), 2),
        ((1, 2, 3), 1),
        ((1,), 0)
    ]
)
def test_take_basic(shape, axis):
    q = get_queue_or_skip()

    Xnp = np.arange(np.prod(shape)).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    indices = dpt.arange(shape[axis])

    Y = dpt.take(X, indices, axis=axis)

    assert_array_equal(Xnp, dpt.asnumpy(Y))


def test_take_axis_arguments():
    q = get_queue_or_skip()

    shape = (4,)
    Xnp = np.arange(shape[0]).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    indices = dpt.arange(shape[0], dtype="int", sycl_queue=q)

    Y = dpt.take(X, indices)
    Ynp = dpt.asnumpy(Y)

    assert_array_equal(Xnp, Ynp)

    X = dpt.reshape(X, (shape[0], 1))
    with pytest.raises(ValueError):
        Y = dpt.take(X, indices)

    with pytest.raises(IndexError):
        Y = dpt.take(X, indices, axis=2)

    indices = dpt.asarray([0], dtype="int", sycl_queue=q)
    Y_1 = dpt.take(X, indices, axis=1)
    Y_2 = dpt.take(X, indices, axis=-1)
    assert_array_equal(dpt.asnumpy(Y_1), dpt.asnumpy(Y_2))


@pytest.mark.parametrize(
    "shape,axis",
    [
        ((4, 3), 0),
        ((5, 5, 5), 2),
        ((1, 2, 3), 1),
        ((1,), 0)
    ]
)
def test_take_numpy(shape, axis):
    q = get_queue_or_skip()

    Xnp = np.arange(np.prod(shape)).reshape(shape)
    X = dpt.asarray(Xnp, sycl_queue=q)

    idx_np = np.asarray([0], dtype="int")
    idx = dpt.asarray(idx_np, sycl_queue=q)

    Ynp = np.take(Xnp, idx_np, axis=axis)
    Y = dpt.take(X, idx, axis=axis)

    assert_array_equal(dpt.asnumpy(Y), Ynp)

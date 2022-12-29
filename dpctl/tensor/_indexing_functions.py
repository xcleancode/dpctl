#                       Data Parallel Control (dpctl)
#
#  Copyright 2020-2022 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import operator

import numpy as np
from numpy.core.numeric import normalize_axis_index

import dpctl
import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as ti


def take(X, indices, /, *, axis=None):

    for i in [X, indices]:
        if not isinstance(i, dpt.usm_ndarray):
            raise TypeError(
                "Expected instance of `dpt.usm_ndarray`, got `{}`.".format(
                    type(i)
                )
            )
    if indices.ndim != 1:
        raise ValueError(
            "`indices` expected array with ndim of 1, got ndim of `{}`".format(
                indices.ndim
            )
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError(
            "`indices` array expected integer data type, got `{}`".format(
                indices.dtype
            )
        )

    X_ndim = X.ndim
    if axis is None:
        if X_ndim > 1:
            raise ValueError(
                "`axis` cannot be `None` for array of dimension `{}`".format(
                    X_ndim
                )
            )
        axis = 0
    else:
        axis = operator.index(axis)
        axis = normalize_axis_index(axis, X_ndim)

    new_shape = X.shape[0:axis] + indices.shape + X.shape[axis+1:]
    exec_q = X.sycl_queue
    res = dpt.empty(
        new_shape, dtype=X.dtype, usm_type=X.usm_type, sycl_queue=exec_q
    )
    arr_index = [np.s_[:]] * X_ndim
    res_index = arr_index.copy()

    hev_list = []
    for i, idx in enumerate(indices):
        res_index[axis], arr_index[axis] = i, idx
        hev, _ = ti._copy_usm_ndarray_into_usm_ndarray(
            X[tuple(arr_index)], res[tuple(res_index)], sycl_queue=exec_q
        )
        hev_list.append(hev)
    dpctl.SyclEvent.wait_for(hev_list)

    return res

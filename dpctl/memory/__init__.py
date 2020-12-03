#===---------- __init__.py - dpctl.memory module -------*- Python -*--------===#
#
#                      Data Parallel Control (dpCtl)
#
# Copyright 2020 Intel Corporation
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
#
#===------------------------------------------------------------------------===#
#
# \file
# This is the dpctl.memory module containing the USM memory manager features
# of dpctl.
#
#===------------------------------------------------------------------------===#
"""
    **Data Parallel Control Memory**

    `dpctl.memory` provides Python objects for untyped USM memory
    container of bytes for each kind of USM pointers: shared pointers,
    device pointers and host pointers.

    Shared and host pointers are accessible from both host and a device,
    while device pointers are only accessible from device.

    Python objects corresponding to shared and host pointers implement
    Python simple buffer protocol. It is therefore possible to use these
    objects to maniputalate USM memory using NumPy or `bytearray`,
    `memoryview`, or `array.array` classes.

"""
from ._memory import MemoryUSMShared, MemoryUSMDevice, MemoryUSMHost
from ._memory import __all__ as _memory__all__

__all__ = _memory__all__

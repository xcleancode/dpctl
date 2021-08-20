//===- dpctl_service.cpp - C API for service functions   -*-C++-*- ===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header defines dpctl service functions.
///
//===----------------------------------------------------------------------===//

#include "dpctl_service.h"
#include "Config/dpctl_config.h"

#include <algorithm>
#include <cstring>
#include <iostream>

__dpctl_give const char *DPCTLService_GetDPCPPVersion(void)
{
    std::string version = DPCTL_DPCPP_VERSION;
    char *version_cstr = nullptr;
    try {
        auto cstr_len = version.length() + 1;
        version_cstr = new char[cstr_len];
#ifdef _WIN32
        strncpy_s(version_cstr, cstr_len, version.c_str(), cstr_len);
#else
        std::strncpy(version_cstr, version.c_str(), cstr_len);
#endif
    } catch (std::bad_alloc const &ba) {
        // \todo log error
        std::cerr << ba.what() << '\n';
    }
    return version_cstr;
}

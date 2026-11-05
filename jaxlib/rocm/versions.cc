/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/rocm/versions_helpers.h"

namespace jax::rocm {
namespace {

namespace nb = nanobind;

NB_MODULE(_versions, m) {
  // Nanobind's leak checking sometimes returns false positives for this file.
  // The problem appears related to forming a closure of a nanobind function.
  nb::set_leak_warnings(false);

  // Build versions, i.e., what version of the headers was JAX compiled against?
  m.def("hip_runtime_build_version", []() { return HIP_VERSION; });

  m.def("hip_runtime_get_version", &HipRuntimeGetVersion);
  m.def("hip_driver_get_version", &HipDriverGetVersion);
  m.def("hip_device_count", &HipDeviceCount);
  m.def("hipdnn_get_version_string", &HipdnnGetVersionString);
  m.def("hipdnn_get_version", &HipdnnGetVersion);
}

}  // namespace
}  // namespace jax::rocm

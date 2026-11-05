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

#include "jaxlib/rocm/versions_helpers.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "absl/base/dynamic_annotations.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "hipdnn_backend.h"

namespace jax::rocm {

int HipRuntimeGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipRuntimeGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int HipDriverGetVersion() {
  int version;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipDriverGetVersion(&version)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&version, sizeof version);
  return version;
}

int HipDeviceCount() {
  int device_count = 0;
  JAX_THROW_IF_ERROR(JAX_AS_STATUS(hipGetDeviceCount(&device_count)));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&device_count, sizeof device_count);
  return device_count;
}

std::string HipdnnGetVersionString() {
  const char* version = hipdnnVersionString_ext();
  if (version == nullptr) {
    throw std::runtime_error("hipDNN not found.");
  }
  return version;
}

int HipdnnGetVersion() {
  const std::string version = HipdnnGetVersionString();
  int major = 0, minor = 0, patch = 0;
  if (std::sscanf(version.c_str(), "%d.%d.%d", &major, &minor, &patch) < 2) {
    throw std::runtime_error("Could not parse hipDNN version '" + version +
                             "'.");
  }
  return major * 10000 + minor * 100 + patch;
}

}  // namespace jax::rocm

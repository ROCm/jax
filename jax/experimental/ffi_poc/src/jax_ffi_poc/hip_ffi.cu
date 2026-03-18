/* Copyright 2024 The JAX Authors.

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

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>
#include <filesystem>
#include "xla/ffi/api/ffi.h"

#include "../kernels/simple_hip_kernel.h"
#include "../shared_obj_handler/aiter_api.h"
#include "../shared_obj_handler/shared_libs.h"

namespace ffi = xla::ffi;


// Get the directory where the current shared library resides
static std::string get_lib_dir() {
    Dl_info info;
    // Use address of any function in the current .so
    if (dladdr(reinterpret_cast<void*>(&get_lib_dir), &info) && info.dli_fname) {
        return std::filesystem::path(info.dli_fname).parent_path().string();
    }
    return ".";
}

using neuron_kernel_fwd_launcher_t =
    hipError_t (*)(hipStream_t, const float*, const float*, float*, float*, size_t);

using neuron_kernel_bwd_launcher_t =
    hipError_t (*)(hipStream_t, const float*, const float*, const float*, float*, float*, size_t);

static ffi::Error HipToFfiError(hipError_t err, const char* where) {
  if (err == hipSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(std::string(where) + ": " + hipGetErrorString(err));
}

#define HIP_RETURN_IF_ERROR(expr)                                      \
  do {                                                                 \
    hipError_t err__ = (expr);                                         \
    if (err__ != hipSuccess) {                                         \
      return ffi::Error::Internal(                                     \
          std::string("HIP error: ") + hipGetErrorString(err__));      \
    }                                                                  \
  } while (0)

extern "C" ffi::Error nk_fwd_wrapper(hipStream_t stream,
                          ffi::Buffer<ffi::F32> a,
                          ffi::Buffer<ffi::F32> b,
                          ffi::ResultBuffer<ffi::F32> c,
                          ffi::ResultBuffer<ffi::F32> b_plus_1,
                          size_t n) {
  static SharedLib lib(get_lib_dir() + "/lib_hip_simple_kernel.so");
  static auto launch_fwd = lib.load<neuron_kernel_fwd_launcher_t>("launch_neuron_kernel_fwd");

  return HipToFfiError(
      launch_fwd(stream, a.typed_data(), b.typed_data(),
                 c->typed_data(), b_plus_1->typed_data(), n),
      "launch_neuron_kernel_fwd");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    neuron_fwd, nk_fwd_wrapper,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});

extern "C" ffi::Error nk_bwd_wrapper(hipStream_t stream,
                          ffi::Buffer<ffi::F32> c_grad,
                          ffi::Buffer<ffi::F32> a,
                          ffi::Buffer<ffi::F32> b_plus_1,
                          ffi::ResultBuffer<ffi::F32> a_grad,
                          ffi::ResultBuffer<ffi::F32> b_grad,
                          size_t n) {
  static SharedLib lib(get_lib_dir() + "/lib_hip_simple_kernel.so");
  static auto launch_bwd = lib.load<neuron_kernel_bwd_launcher_t>("launch_neuron_kernel_bwd");

  return HipToFfiError(
      launch_bwd(stream, c_grad.typed_data(), a.typed_data(), b_plus_1.typed_data(),
                 a_grad->typed_data(), b_grad->typed_data(), n),
      "launch_neuron_kernel_bwd");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    neuron_bwd, nk_bwd_wrapper,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});
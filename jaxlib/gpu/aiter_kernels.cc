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

#include "jaxlib/gpu/aiter_kernels.h"

#include <string>

#include "jaxlib/gpu/aiter_kernels.cu.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {

namespace ffi = xla::ffi;

static ffi::Error GpuToFfiError(gpuError_t err, const char* where) {
  if (err == gpuSuccess) return ffi::Error::Success();
  return ffi::Error::Internal(
      std::string(where) + ": " + gpuGetErrorString(err));
}

static ffi::Error NeuronFwdImpl(gpuStream_t stream,
                                ffi::Buffer<ffi::F32> a,
                                ffi::Buffer<ffi::F32> b,
                                ffi::ResultBuffer<ffi::F32> c,
                                ffi::ResultBuffer<ffi::F32> b_plus_1,
                                size_t n) {
  return GpuToFfiError(
      LaunchNeuronKernelFwd(stream, a.typed_data(), b.typed_data(),
                            c->typed_data(), b_plus_1->typed_data(), n),
      "LaunchNeuronKernelFwd");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NeuronForwardFfi, NeuronFwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});



static ffi::Error NeuronBwdImpl(gpuStream_t stream,
                                ffi::Buffer<ffi::F32> c_grad,
                                ffi::Buffer<ffi::F32> a,
                                ffi::Buffer<ffi::F32> b_plus_1,
                                ffi::ResultBuffer<ffi::F32> a_grad,
                                ffi::ResultBuffer<ffi::F32> b_grad,
                                size_t n) {
  return GpuToFfiError(
      LaunchNeuronKernelBwd(stream, c_grad.typed_data(), a.typed_data(),
                            b_plus_1.typed_data(), a_grad->typed_data(),
                            b_grad->typed_data(), n),
      "LaunchNeuronKernelBwd");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    NeuronBackwardFfi, NeuronBwdImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<gpuStream_t>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Arg<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Ret<ffi::Buffer<ffi::F32>>()
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

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

#include "jaxlib/gpu/aiter_mha.h"

#include <algorithm>
#include <cstddef>

#include "jaxlib/gpu/vendor.h"

namespace jax {
namespace JAX_GPU_NAMESPACE {
namespace {

__global__ void NeuronFwdKernel(const float* a, const float* b, float* c,
                                float* b_plus_1, size_t n) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    float bp1 = b[idx] + 1.0f;
    b_plus_1[idx] = bp1;
    c[idx] = a[idx] * bp1;
  }
}

__global__ void NeuronBwdKernel(const float* c_grad, const float* a,
                                const float* b_plus_1, float* a_grad,
                                float* b_grad, size_t n) {
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += blockDim.x * gridDim.x) {
    a_grad[idx] = c_grad[idx] * b_plus_1[idx];
    b_grad[idx] = c_grad[idx] * a[idx];
  }
}

}  // namespace

gpuError_t LaunchNeuronKernelFwd(gpuStream_t stream, const float* a,
                                 const float* b, float* c, float* b_plus_1,
                                 size_t n) {
  constexpr int kBlockSize = 256;
  int num_blocks =
      std::min<int>((n + kBlockSize - 1) / kBlockSize, 1024);
  NeuronFwdKernel<<<num_blocks, kBlockSize, 0, stream>>>(a, b, c, b_plus_1, n);
  return gpuGetLastError();
}

gpuError_t LaunchNeuronKernelBwd(gpuStream_t stream, const float* c_grad,
                                 const float* a, const float* b_plus_1,
                                 float* a_grad, float* b_grad, size_t n) {
  constexpr int kBlockSize = 256;
  int num_blocks =
      std::min<int>((n + kBlockSize - 1) / kBlockSize, 1024);
  NeuronBwdKernel<<<num_blocks, kBlockSize, 0, stream>>>(c_grad, a, b_plus_1,
                                                          a_grad, b_grad, n);
  return gpuGetLastError();
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

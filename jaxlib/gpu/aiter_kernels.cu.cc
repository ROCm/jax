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

#include "jaxlib/gpu/aiter_kernels.cu.h"
#include "jaxlib/gpu/vendor.h"

// c = a * (b + 1)
__global__ void neuron_kernel_fwd(const float* a, const float* b,
                                  float* c, float* b_plus_1, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}

// da = dc * (b+1),  db = dc * a
__global__ void neuron_kernel_bwd(const float* c_grad, const float* a,
                                  const float* b_plus_1, float* a_grad,
                                  float* b_grad, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

namespace jax {
namespace JAX_GPU_NAMESPACE {

gpuError_t LaunchNeuronKernelFwd(gpuStream_t stream,
                                 const float* a, const float* b,
                                 float* c, float* b_plus_1, size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);
  if (grid_dim < 1) grid_dim = 1;
  neuron_kernel_fwd<<<grid_dim, block_dim, 0, stream>>>(a, b, c, b_plus_1, n);
  return gpuGetLastError();
}

gpuError_t LaunchNeuronKernelBwd(gpuStream_t stream,
                                 const float* c_grad, const float* a,
                                 const float* b_plus_1, float* a_grad,
                                 float* b_grad, size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);
  if (grid_dim < 1) grid_dim = 1;
  neuron_kernel_bwd<<<grid_dim, block_dim, 0, stream>>>(
      c_grad, a, b_plus_1, a_grad, b_grad, n);
  return gpuGetLastError();
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax

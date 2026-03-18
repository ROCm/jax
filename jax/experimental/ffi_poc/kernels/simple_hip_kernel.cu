

#include "simple_hip_kernel.h"


//----------------------------------------------------------------------------//
//                            Forward pass                                    //
//----------------------------------------------------------------------------//

// c = a * (b+1)
// This strawman operation works well for demo purposes because:
// 1. it's simple enough to be quickly understood,
// 2. it's complex enough to require intermediate outputs in grad computation,
//    like many operations in practice do, and
// 3. it does not have a built-in implementation in JAX.
__global__ void neuron_kernel_fwd(const float *a, const float *b, float *c,
                             float *b_plus_1,  // intermediate output b+1
                             size_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}



//----------------------------------------------------------------------------//
//                            Backward pass                                   //
//----------------------------------------------------------------------------//

// compute da = dc * (b+1), and
//         db = dc * a
__global__ void neuron_kernel_bwd(const float *c_grad,    // incoming gradient wrt c
                             const float *a,         // original input a
                             const float *b_plus_1,  // intermediate output b+1
                             float *a_grad,          // outgoing gradient wrt a
                             float *b_grad,          // outgoing gradient wrt b
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    // In practice on GPUs b_plus_1 can be recomputed for practically free
    // instead of storing it out and reusing, so the reuse here is a bit
    // contrived. We do it to demonstrate residual/intermediate output passing
    // between the forward and the backward pass which becomes useful when
    // recomputation is more expensive than reuse.
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}


extern "C" hipError_t launch_neuron_kernel_fwd(hipStream_t stream,
                                               const float* a,
                                               const float* b,
                                               float* c,
                                               float* b_plus_1,
                                               size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);
  if (grid_dim < 1) grid_dim = 1;

  hipLaunchKernelGGL(neuron_kernel_fwd,
                     dim3(grid_dim),
                     dim3(block_dim),
                     0,
                     stream,
                     a, b, c, b_plus_1, n);
  return hipGetLastError();
}

extern "C" hipError_t launch_neuron_kernel_bwd(hipStream_t stream,
                                               const float* c_grad,
                                               const float* a,
                                               const float* b_plus_1,
                                               float* a_grad,
                                               float* b_grad,
                                               size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);
  if (grid_dim < 1) grid_dim = 1;

  hipLaunchKernelGGL(neuron_kernel_bwd,
                     dim3(grid_dim),
                     dim3(block_dim),
                     0,
                     stream,
                     c_grad, a, b_plus_1, a_grad, b_grad, n);
  return hipGetLastError();
}
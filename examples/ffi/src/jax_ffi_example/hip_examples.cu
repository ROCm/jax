#include <hip/hip_runtime.h>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

//----------------------------------------------------------------------------//
//                            Forward pass                                    //
//----------------------------------------------------------------------------//

// c = a * (b + 1)
__global__ void FooFwdKernel(const float* a,
                             const float* b,
                             float* c,
                             float* b_plus_1,
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid_stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}

ffi::Error FooFwdHost(hipStream_t stream,
                      ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b,
                      ffi::ResultBuffer<ffi::F32> c,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);

  // Optional clamp to avoid absurdly large grids
  if (grid_dim < 1) grid_dim = 1;

  hipLaunchKernelGGL(FooFwdKernel,
                     dim3(grid_dim),
                     dim3(block_dim),
                     0,
                     stream,
                     a.typed_data(),
                     b.typed_data(),
                     c->typed_data(),
                     b_plus_1->typed_data(),
                     n);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error::Internal(
        std::string("HIP error launching FooFwdKernel: ") +
        hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooFwd, FooFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b
        .Ret<ffi::Buffer<ffi::F32>>()             // c
        .Ret<ffi::Buffer<ffi::F32>>()             // b_plus_1
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});

//----------------------------------------------------------------------------//
//                            Backward pass                                   //
//----------------------------------------------------------------------------//

// da = dc * (b + 1)
// db = dc * a
__global__ void FooBwdKernel(const float* c_grad,
                             const float* a,
                             const float* b_plus_1,
                             float* a_grad,
                             float* b_grad,
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t grid_stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += grid_stride) {
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

ffi::Error FooBwdHost(hipStream_t stream,
                      ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      size_t n) {
  constexpr int block_dim = 128;
  int grid_dim = static_cast<int>((n + block_dim - 1) / block_dim);

  if (grid_dim < 1) grid_dim = 1;

  hipLaunchKernelGGL(FooBwdKernel,
                     dim3(grid_dim),
                     dim3(block_dim),
                     0,
                     stream,
                     c_grad.typed_data(),
                     a.typed_data(),
                     b_plus_1.typed_data(),
                     a_grad->typed_data(),
                     b_grad->typed_data(),
                     n);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error::Internal(
        std::string("HIP error launching FooBwdKernel: ") +
        hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooBwd, FooBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // c_grad
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b_plus_1
        .Ret<ffi::Buffer<ffi::F32>>()             // a_grad
        .Ret<ffi::Buffer<ffi::F32>>()             // b_grad
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});
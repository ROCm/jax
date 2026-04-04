// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "logging.h"
#include "mha_common_utils.h"
#include <chrono>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifdef __HIP_PLATFORM_AMD__
typedef hip_bfloat16 hip_bf16_type;
#else
typedef __hip_bfloat16 hip_bf16_type;
#endif

namespace jax_aiter {
namespace mha_utils {

template <typename T>
__global__ void mqa_gqa_reduce_kernel(const T *__restrict__ d_expanded,
                                      T *__restrict__ d_reduced,
                                      int64_t batch_size, int64_t seqlen,
                                      int64_t num_heads_q, int64_t num_heads_k,
                                      int64_t head_dim, int64_t num_groups) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * seqlen * num_heads_k * head_dim;

  if (tid >= total_elements)
    return;

  int b = tid / (seqlen * num_heads_k * head_dim);
  int remainder = tid % (seqlen * num_heads_k * head_dim);
  int s = remainder / (num_heads_k * head_dim);
  remainder = remainder % (num_heads_k * head_dim);
  int hk = remainder / head_dim;
  int d = remainder % head_dim;

  float sum = 0.0f;
  for (int g = 0; g < num_groups; g++) {
    int h = hk * num_groups + g;
    int expanded_idx = b * seqlen * num_heads_q * head_dim +
                       s * num_heads_q * head_dim + h * head_dim + d;
    sum += static_cast<float>(d_expanded[expanded_idx]);
  }

  d_reduced[tid] = static_cast<T>(sum);
}

template __global__ void
mqa_gqa_reduce_kernel<__half>(const __half *__restrict__, __half *__restrict__,
                              int64_t, int64_t, int64_t, int64_t, int64_t,
                              int64_t);
template __global__ void mqa_gqa_reduce_kernel<hip_bf16_type>(
    const hip_bf16_type *__restrict__, hip_bf16_type *__restrict__, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t);

JAX_AITER_EXPORT
void launch_mqa_gqa_reduction(const void *src, void *dst, int64_t batch_size,
                              int64_t seqlen_k, int64_t num_heads_q,
                              int64_t num_heads_k, int64_t head_size,
                              int64_t groups, xla::ffi::DataType dtype,
                              hipStream_t stream) {

  int64_t total_elements = batch_size * seqlen_k * num_heads_k * head_size;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;

  if (dtype == xla::ffi::DataType::F16) {
    mqa_gqa_reduce_kernel<__half><<<blocks, threads, 0, stream>>>(
        static_cast<const __half *>(src), static_cast<__half *>(dst),
        batch_size, seqlen_k, num_heads_q, num_heads_k, head_size, groups);
  } else if (dtype == xla::ffi::DataType::BF16) {
    mqa_gqa_reduce_kernel<hip_bf16_type><<<blocks, threads, 0, stream>>>(
        static_cast<const hip_bf16_type *>(src),
        static_cast<hip_bf16_type *>(dst), batch_size, seqlen_k, num_heads_q,
        num_heads_k, head_size, groups);
  }
}

JAX_AITER_EXPORT
xla::ffi::Error
prepare_rng_state_for_fwd(hipStream_t stream, float dropout_p, int dev_idx,
                          int64_t batch_size, int64_t num_heads,
                          const std::optional<xla::ffi::AnyBuffer> &gen,
                          xla::ffi::Result<xla::ffi::AnyBuffer> &rng_state,
                          RngStatePointers &out_ptrs) {

  // Ensure rng_state buffer is valid
  if (rng_state->size_bytes() < 2 * sizeof(int64_t)) {
    return xla::ffi::Error(
        xla::ffi::ErrorCode::kInvalidArgument,
        "rng_state result buffer must have at least 2 int64s");
  }

  uint64_t *rng_state_ptr =
      reinterpret_cast<uint64_t *>(rng_state->untyped_data());

  if (dropout_p > 0.0f) {
    uint64_t seed_value, offset_value;

    if (gen.has_value() && gen->size_bytes() >= 2 * sizeof(int64_t)) {
      const auto *gen_data = static_cast<const int64_t *>(gen->untyped_data());
      seed_value = static_cast<uint64_t>(gen_data[0]);
      offset_value = static_cast<uint64_t>(gen_data[1]);
      JA_LOG("[JAX_AITER_CPP] Using provided generator with seed: %llu, "
             "offset: %llu",
             seed_value, offset_value);

      hipError_t err = hipMemcpyAsync(
          rng_state->untyped_data(), gen_data, 2 * sizeof(int64_t),
          hipMemcpyDeviceToDevice, // Try device-to-device first
          stream);

      if (err == hipErrorInvalidValue) {
        err =
            hipMemcpyAsync(rng_state->untyped_data(), gen_data,
                           2 * sizeof(int64_t), hipMemcpyHostToDevice, stream);
      }

      if (err != hipSuccess) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            std::string("Failed to copy RNG state to device: ") +
                hipGetErrorString(err));
      }
    } else {
      auto now = std::chrono::high_resolution_clock::now();
      auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(
                           now.time_since_epoch())
                           .count();

      seed_value =
          static_cast<uint64_t>(timestamp) ^ static_cast<uint64_t>(dev_idx);
      offset_value = static_cast<uint64_t>(batch_size * num_heads *
                                           ck_tile::get_warp_size());

      JA_LOG("[JAX_AITER_CPP] Generated RNG with seed: %llu, offset: %llu (no "
             "gen_ provided)",
             seed_value, offset_value);

      uint64_t host_rng[2] = {seed_value, offset_value};
      hipError_t err =
          hipMemcpyAsync(rng_state->untyped_data(), host_rng,
                         2 * sizeof(int64_t), hipMemcpyHostToDevice, stream);

      if (err != hipSuccess) {
        return xla::ffi::Error(
            xla::ffi::ErrorCode::kInternal,
            std::string("Failed to copy generated RNG state to device: ") +
                hipGetErrorString(err));
      }
    }

    out_ptrs.seed = rng_state_ptr;
    out_ptrs.offset = rng_state_ptr + 1;
  } else {
    hipError_t err = hipMemsetAsync(rng_state->untyped_data(), 0,
                                    2 * sizeof(int64_t), stream);
    if (err != hipSuccess) {
      JA_LOG("[JAX_AITER_CPP] Warning: Failed to zero RNG state: %s",
             hipGetErrorString(err));
    }

    out_ptrs.seed = rng_state_ptr;
    out_ptrs.offset = rng_state_ptr + 1;
  }

  return xla::ffi::Error::Success();
}

} // namespace mha_utils
} // namespace jax_aiter

// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <string>
#include <variant>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "ck_tile/ops/bias.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/mask.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define JAX_AITER_EXPORT __attribute__((visibility("default")))
#else
#define JAX_AITER_EXPORT
#endif

namespace jax_aiter {
namespace mha_utils {

inline std::string dtype_to_string(xla::ffi::DataType dtype) {
  switch (dtype) {
  case xla::ffi::DataType::F16:
    return "fp16";
  case xla::ffi::DataType::BF16:
    return "bf16";
  case xla::ffi::DataType::F32:
    return "fp32";
  default:
    throw std::runtime_error("Unsupported dtype for MHA");
  }
}

inline size_t dtype_size(xla::ffi::DataType dtype) {
  switch (dtype) {
  case xla::ffi::DataType::F16:
  case xla::ffi::DataType::BF16:
    return 2;
  case xla::ffi::DataType::F32:
    return 4;
  case xla::ffi::DataType::U8:
    return 1;
  case xla::ffi::DataType::S32:
    return 4;
  case xla::ffi::DataType::S64:
    return 8;
  default:
    return 4;
  }
}

inline ck_tile::index_t
calculate_stride(const xla::ffi::Span<const int64_t> &dims, int dim_idx) {
  if (dim_idx >= dims.size())
    return 0;
  ck_tile::index_t stride = 1;
  for (int i = dims.size() - 1; i > dim_idx; --i) {
    stride *= dims[i];
  }
  return stride;
}

inline mask_info create_mask_info(bool is_causal, int window_size_left,
                                  int window_size_right,
                                  ck_tile::index_t seqlen_q,
                                  ck_tile::index_t seqlen_k) {
  mask_info mask;
  if (is_causal) {
    window_size_right = 0;
    std::string mask_identify = "b:" + std::to_string(window_size_left) + ",0";
    mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k);
  } else if (window_size_left == -1 && window_size_right == -1) {
    mask = mask_info::decode("0", seqlen_q, seqlen_k);
  } else {
    std::string mask_identify = "b:" + std::to_string(window_size_left) + "," +
                                std::to_string(window_size_right);
    mask = mask_info::decode(mask_identify, seqlen_q, seqlen_k);
  }
  return mask;
}

inline bias_enum get_bias_type(bool has_bias, bool has_alibi) {
  if (has_alibi)
    return bias_enum::alibi;
  if (has_bias)
    return bias_enum::elementwise_bias;
  return bias_enum::no_bias;
}

inline float calculate_p_undrop(float dropout_p) {
  return (dropout_p > 0.0f) ? (1.0f - dropout_p) : 1.0f;
}

inline ck_tile::stream_config create_stream_config(hipStream_t stream,
                                                   bool log_enable = false) {
  return ck_tile::stream_config{stream, false, log_enable};
}

inline bool is_valid_buffer(const xla::ffi::AnyBuffer &buffer) {
  return buffer.untyped_data() != nullptr && buffer.size_bytes() > 0;
}

inline void
validate_mha_dimensions(const xla::ffi::Span<const int64_t> &q_dims,
                        const xla::ffi::Span<const int64_t> &k_dims,
                        const xla::ffi::Span<const int64_t> &v_dims) {

  if (q_dims.size() != 4 || k_dims.size() != 4 || v_dims.size() != 4) {
    throw std::runtime_error(
        "Q, K, V must be 4D tensors [batch, seqlen, nheads, hdim]");
  }
  if (q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0]) {
    throw std::runtime_error("Batch sizes must match for Q, K, V");
  }
  if (q_dims[3] != k_dims[3]) {
    throw std::runtime_error("Head dimensions must match for Q and K");
  }
  if (k_dims[1] != v_dims[1] || k_dims[2] != v_dims[2]) {
    throw std::runtime_error(
        "Sequence length and num_heads must match for K and V");
  }
}

inline std::pair<uint64_t *, uint64_t *> get_rng_seed_offset_ptrs(
    const std::optional<xla::ffi::AnyBuffer> &rng_state_opt, float dropout_p) {

  if (dropout_p == 0.0f) {
    return {nullptr, nullptr};
  }

  if (!rng_state_opt.has_value() || !is_valid_buffer(*rng_state_opt)) {
    throw std::runtime_error("rng_state must be provided when dropout_p > 0");
  }

  const auto &rng_state = *rng_state_opt;
  if (rng_state.size_bytes() < 2 * sizeof(uint64_t)) {
    throw std::runtime_error(
        "rng_state buffer too small, need at least 16 bytes");
  }

  auto *base_ptr =
      static_cast<uint64_t *>(const_cast<void *>(rng_state.untyped_data()));
  return {base_ptr, base_ptr + 1};
}

inline bool is_mqa_gqa(int64_t num_heads_q, int64_t num_heads_k,
                       int64_t *groups_out = nullptr) {
  bool is_mqa = (num_heads_q != num_heads_k);
  if (is_mqa && groups_out != nullptr) {
    if (num_heads_q % num_heads_k != 0) {
      throw std::runtime_error(
          "num_heads_q must be divisible by num_heads_k for MQA/GQA");
    }
    *groups_out = num_heads_q / num_heads_k;
  }
  return is_mqa;
}

inline void validate_mha_bwd_inputs(const xla::ffi::AnyBuffer &dout,
                                    const xla::ffi::AnyBuffer &q,
                                    const xla::ffi::AnyBuffer &k,
                                    const xla::ffi::AnyBuffer &v,
                                    const xla::ffi::AnyBuffer &out,
                                    const xla::ffi::AnyBuffer &softmax_lse,
                                    int64_t head_size_q, int64_t head_size_v,
                                    int64_t num_heads_q, int64_t num_heads_k) {

  auto q_dtype = q.element_type();
  if (q_dtype != xla::ffi::DataType::F16 &&
      q_dtype != xla::ffi::DataType::BF16) {
    throw std::runtime_error(
        "FlashAttention only supports fp16/bf16 data type");
  }

  if (k.element_type() != q_dtype || v.element_type() != q_dtype ||
      out.element_type() != q_dtype || dout.element_type() != q_dtype) {
    throw std::runtime_error("Q, K, V, OUT, DOUT must have the same dtype");
  }

  if (softmax_lse.element_type() != xla::ffi::DataType::F32) {
    throw std::runtime_error("softmax_lse must be float32");
  }

  if (head_size_q % 8 != 0 || head_size_v % 8 != 0) {
    throw std::runtime_error(
        "head_size_q and head_size_v must be multiples of 8");
  }

  if (head_size_q > 256 || head_size_v > 256) {
    throw std::runtime_error(
        "CK FlashAttention backward only supports head dimension at most 256");
  }

  if (num_heads_q % num_heads_k != 0) {
    throw std::runtime_error(
        "Number of heads in query must divide number of heads in key/value");
  }
}

JAX_AITER_EXPORT
void launch_mqa_gqa_reduction(const void *src, void *dst, int64_t batch_size,
                              int64_t seqlen_k, int64_t num_heads_q,
                              int64_t num_heads_k, int64_t head_size,
                              int64_t groups, xla::ffi::DataType dtype,
                              hipStream_t stream);

// Holds device pointers to seed and offset for RNG state.
struct RngStatePointers {
  uint64_t *seed;
  uint64_t *offset;
};

// Prepares RNG state for forward pass: uses gen_ if provided, else generates
// internally.
JAX_AITER_EXPORT
xla::ffi::Error
prepare_rng_state_for_fwd(hipStream_t stream, float dropout_p, int dev_idx,
                          int64_t batch_size, int64_t num_heads,
                          const std::optional<xla::ffi::AnyBuffer> &gen,
                          xla::ffi::Result<xla::ffi::AnyBuffer> &rng_state,
                          RngStatePointers &out_ptrs);

} // namespace mha_utils
} // namespace jax_aiter

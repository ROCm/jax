// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Unified MHA forward FFI handler for both batch and varlen modes.
// Detects mode from tensor rank: 4D = batch [b,s,h,d], 3D = varlen [total,h,d].
// Calls aiter::mha_fwd(args, stream) which handles CK vs ASM v3 internally.

#include <chrono>
#include <cstring>
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter/mha_fwd.h"
#include "mha_common_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error
aiter_mha_fwd_impl(
    hipStream_t stream,
    ffi::AnyBuffer q,
    ffi::AnyBuffer k,
    ffi::AnyBuffer v,
    std::optional<ffi::AnyBuffer> cu_seqlens_q_,
    std::optional<ffi::AnyBuffer> cu_seqlens_kv_,
    std::optional<ffi::AnyBuffer> out_,
    std::optional<ffi::AnyBuffer> bias_,
    std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> gen_,
    ffi::Result<ffi::AnyBuffer> o,
    ffi::Result<ffi::AnyBuffer> lse,
    ffi::Result<ffi::AnyBuffer> p,
    ffi::Result<ffi::AnyBuffer> rng_state,
    float dropout_p,
    float softmax_scale,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    bool return_softmax_lse,
    bool return_dropout_randval,
    bool use_asm_v3,
    int how_v3_bf16_cvt,
    int max_seqlen_q_attr,
    int max_seqlen_k_attr,
    int min_seqlen_q,
    float logits_soft_cap,
    bool zero_tensors) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v) is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");
  }

  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();

  // Detect batch (4D) vs varlen (3D) from tensor rank.
  const bool is_varlen = (q_dims.size() == 3);

  int64_t batch_size, seqlen_q, num_heads, head_size_q;
  int64_t seqlen_k, num_heads_k, head_size_v;
  int64_t max_seqlen_q, max_seqlen_k;

  if (is_varlen) {
    // [total_q, nheads, hdim]
    int64_t total_q = q_dims[0];
    num_heads = q_dims[1];
    head_size_q = q_dims[2];
    int64_t total_k = k_dims[0];
    num_heads_k = k_dims[1];
    head_size_v = v_dims[2];

    if (!cu_seqlens_q_.has_value() || !mha_utils::is_valid_buffer(*cu_seqlens_q_)) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "varlen mode requires cu_seqlens_q");
    }
    batch_size = cu_seqlens_q_->dimensions()[0] - 1;
    seqlen_q = total_q;
    seqlen_k = total_k;
    max_seqlen_q = max_seqlen_q_attr;
    max_seqlen_k = max_seqlen_k_attr;
  } else {
    // [batch, seqlen, nheads, hdim]
    mha_utils::validate_mha_dimensions(q_dims, k_dims, v_dims);
    batch_size = q_dims[0];
    seqlen_q = q_dims[1];
    num_heads = q_dims[2];
    head_size_q = q_dims[3];
    seqlen_k = k_dims[1];
    num_heads_k = k_dims[2];
    head_size_v = v_dims[3];
    max_seqlen_q = seqlen_q;
    max_seqlen_k = seqlen_k;
  }

  auto q_dtype = q.element_type();
  if (q_dtype != ffi::DataType::F16 && q_dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FlashAttention only supports fp16 and bf16");
  }
  if (k.element_type() != q_dtype || v.element_type() != q_dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "q, k, v must have the same dtype");
  }
  if (batch_size <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "batch size must be positive");
  }
  if (head_size_q > 256 || head_size_v > 256) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "head dim must be <= 256");
  }
  if (head_size_q % 8 != 0 || head_size_v % 8 != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "head dim must be multiple of 8");
  }
  if (num_heads % num_heads_k != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "num_heads_q must be divisible by num_heads_k");
  }
  if (return_dropout_randval && dropout_p <= 0.0f) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "return_dropout_randval requires dropout_p > 0");
  }

  std::string dtype_str = mha_utils::dtype_to_string(q_dtype);

  // Bias / ALiBi handling.
  const void *bias_ptr = nullptr;
  ck_tile::index_t stride_bias = 0;
  bool has_bias = bias_.has_value() && mha_utils::is_valid_buffer(*bias_);
  bool has_alibi = alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_);

  if (has_bias && has_alibi) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "cannot apply both bias and alibi");
  }
  if (has_bias) {
    bias_ptr = bias_->untyped_data();
    auto bd = bias_->dimensions();
    stride_bias = bd.size() >= 2 ? mha_utils::calculate_stride(bd, 0) : 0;
  } else if (has_alibi) {
    bias_ptr = alibi_slopes_->untyped_data();
    auto ad = alibi_slopes_->dimensions();
    stride_bias = ad.size() >= 2 ? mha_utils::calculate_stride(ad, 0) : 0;
  }
  bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);

  // Causal / window normalization.
  int w_left = window_size_left;
  int w_right = window_size_right;
  int ref_sk = is_varlen ? max_seqlen_k : seqlen_k;
  int ref_sq = is_varlen ? max_seqlen_q : seqlen_q;
  if (w_left >= ref_sk) w_left = -1;
  if (w_right >= ref_sk) w_right = -1;

  bool eff_causal = is_causal;
  if (ref_sq == 1 && !has_alibi) eff_causal = false;

  mask_info mask = mha_utils::create_mask_info(eff_causal, w_left, w_right, ref_sq, ref_sk);

  // Zero tensors (varlen).
  if (zero_tensors) {
    HIP_CHECK(hipMemsetAsync(o->untyped_data(), 0, o->size_bytes(), stream));
    if (return_softmax_lse && lse->size_bytes() > 0) {
      size_t n = lse->element_count();
      std::vector<float> neg_inf(n, -std::numeric_limits<float>::infinity());
      HIP_CHECK(hipMemcpyAsync(lse->untyped_data(), neg_inf.data(),
                               n * sizeof(float), hipMemcpyHostToDevice, stream));
    }
    if (return_dropout_randval && p->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(p->untyped_data(), 0, p->size_bytes(), stream));
    }
  }

  // Empty key fast path.
  if (ref_sk == 0) {
    HIP_CHECK(hipMemsetAsync(o->untyped_data(), 0, o->size_bytes(), stream));
    if (return_softmax_lse && lse->size_bytes() > 0) {
      size_t n = lse->element_count();
      std::vector<float> inf_buf(n, std::numeric_limits<float>::infinity());
      HIP_CHECK(hipMemcpyAsync(lse->untyped_data(), inf_buf.data(),
                               n * sizeof(float), hipMemcpyHostToDevice, stream));
    }
    return ffi::Error::Success();
  }

  // RNG state for dropout.
  mha_utils::RngStatePointers rng_ptrs;
  auto rng_err = mha_utils::prepare_rng_state_for_fwd(
      stream, dropout_p, dev_idx, batch_size, num_heads, gen_, rng_state, rng_ptrs);
  if (!rng_err.success()) return rng_err;

  // Sequence length pointers.
  const ck_tile::index_t *seqstart_q_ptr = nullptr;
  const ck_tile::index_t *seqstart_k_ptr = nullptr;
  const ck_tile::index_t *cu_seqlen_q_ptr = nullptr;
  const ck_tile::index_t *cu_seqlen_k_ptr = nullptr;

  if (is_varlen) {
    seqstart_q_ptr = reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_q_->untyped_data());
    if (cu_seqlens_kv_.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_kv_)) {
      seqstart_k_ptr = reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_kv_->untyped_data());
    }
  } else {
    if (cu_seqlens_q_.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_q_)) {
      cu_seqlen_q_ptr = reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_q_->untyped_data());
    }
    if (cu_seqlens_kv_.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_kv_)) {
      cu_seqlen_k_ptr = reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_kv_->untyped_data());
    }
  }

  // Compute strides based on tensor rank.
  auto o_dims = o->dimensions();
  auto lse_dims = lse->dimensions();
  auto p_dims = p->dimensions();

  ck_tile::index_t stride_q, stride_k, stride_v, stride_o;
  ck_tile::index_t nhead_stride_q, nhead_stride_k, nhead_stride_v, nhead_stride_o;
  ck_tile::index_t batch_stride_q = 0, batch_stride_k = 0, batch_stride_v = 0, batch_stride_o = 0;

  if (is_varlen) {
    // 3D: [total, nheads, d] → stride at dim 0, nhead_stride at dim 1
    stride_q = mha_utils::calculate_stride(q_dims, 0);
    stride_k = mha_utils::calculate_stride(k_dims, 0);
    stride_v = mha_utils::calculate_stride(v_dims, 0);
    stride_o = mha_utils::calculate_stride(o_dims, 0);
    nhead_stride_q = mha_utils::calculate_stride(q_dims, 1);
    nhead_stride_k = mha_utils::calculate_stride(k_dims, 1);
    nhead_stride_v = mha_utils::calculate_stride(v_dims, 1);
    nhead_stride_o = mha_utils::calculate_stride(o_dims, 1);
  } else {
    // 4D: [batch, seqlen, nheads, d]
    stride_q = mha_utils::calculate_stride(q_dims, 1);
    stride_k = mha_utils::calculate_stride(k_dims, 1);
    stride_v = mha_utils::calculate_stride(v_dims, 1);
    stride_o = mha_utils::calculate_stride(o_dims, 1);
    nhead_stride_q = mha_utils::calculate_stride(q_dims, 2);
    nhead_stride_k = mha_utils::calculate_stride(k_dims, 2);
    nhead_stride_v = mha_utils::calculate_stride(v_dims, 2);
    nhead_stride_o = mha_utils::calculate_stride(o_dims, 2);
    batch_stride_q = mha_utils::calculate_stride(q_dims, 0);
    batch_stride_k = mha_utils::calculate_stride(k_dims, 0);
    batch_stride_v = mha_utils::calculate_stride(v_dims, 0);
    batch_stride_o = mha_utils::calculate_stride(o_dims, 0);
  }

  ck_tile::index_t nhead_stride_lse = 0, batch_stride_lse = 0;
  if (return_softmax_lse && lse_dims.size() >= 2) {
    if (is_varlen) {
      nhead_stride_lse = mha_utils::calculate_stride(lse_dims, 0);
    } else {
      nhead_stride_lse = mha_utils::calculate_stride(lse_dims, 1);
      batch_stride_lse = mha_utils::calculate_stride(lse_dims, 0);
    }
  }

  ck_tile::index_t stride_randval = 0, nhead_stride_randval = 0, batch_stride_randval = 0;
  if (return_dropout_randval && p_dims.size() >= 2) {
    if (is_varlen) {
      stride_randval = mha_utils::calculate_stride(p_dims, 1);
      nhead_stride_randval = mha_utils::calculate_stride(p_dims, 0);
    } else {
      stride_randval = mha_utils::calculate_stride(p_dims, 2);
      nhead_stride_randval = mha_utils::calculate_stride(p_dims, 1);
      batch_stride_randval = mha_utils::calculate_stride(p_dims, 0);
    }
  }

  if (return_dropout_randval && !is_varlen && p->size_bytes() > 0) {
    HIP_CHECK(hipMemsetAsync(p->untyped_data(), 0, p->size_bytes(), stream));
  }

  auto args = aiter::mha_fwd_args{
      .use_asm_v3 = use_asm_v3,
      .v3_api_check = false,
      .how_v3_bf16_cvt = how_v3_bf16_cvt,
      .data_type = dtype_str,
      .is_group_mode = is_varlen,
      .bias_type = static_cast<int>(bias_type),
      .has_lse = return_softmax_lse,
      .qscale_type = 0,
      .has_sink = false,

      .q_ptr = q.untyped_data(),
      .k_ptr = k.untyped_data(),
      .v_ptr = v.untyped_data(),
      .bias_ptr = bias_ptr,
      .q_descale_ptr = nullptr,
      .k_descale_ptr = nullptr,
      .v_descale_ptr = nullptr,
      .rand_val_ptr = return_dropout_randval ? p->untyped_data() : nullptr,
      .lse_ptr = return_softmax_lse ? lse->untyped_data() : nullptr,
      .o_ptr = o->untyped_data(),

      .seqstart_q_ptr = seqstart_q_ptr,
      .seqstart_k_ptr = seqstart_k_ptr,
      .seqlen_q_ptr = nullptr,
      .seqlen_k_ptr = nullptr,
      .cu_seqlen_q_ptr = cu_seqlen_q_ptr,
      .cu_seqlen_k_ptr = cu_seqlen_k_ptr,
      .block_scale_seqstart_q_ptr = nullptr,
      .block_scale_seqstart_k_ptr = nullptr,
      .sink_ptr = nullptr,

      .seqlen_q = static_cast<ck_tile::index_t>(seqlen_q),
      .seqlen_k = static_cast<ck_tile::index_t>(seqlen_k),
      .batch = static_cast<ck_tile::index_t>(batch_size),
      .max_seqlen_q = static_cast<ck_tile::index_t>(max_seqlen_q),
      .hdim_q = static_cast<ck_tile::index_t>(head_size_q),
      .hdim_v = static_cast<ck_tile::index_t>(head_size_v),
      .nhead_q = static_cast<ck_tile::index_t>(num_heads),
      .nhead_k = static_cast<ck_tile::index_t>(num_heads_k),

      .scale_s = softmax_scale,
      .logits_soft_cap = logits_soft_cap,

      .stride_q = stride_q,
      .stride_k = stride_k,
      .stride_v = stride_v,
      .stride_bias = stride_bias,
      .stride_randval = stride_randval,
      .stride_o = stride_o,
      .nhead_stride_q = nhead_stride_q,
      .nhead_stride_k = nhead_stride_k,
      .nhead_stride_v = nhead_stride_v,
      .nhead_stride_bias = 0,
      .nhead_stride_randval = nhead_stride_randval,
      .nhead_stride_lse = nhead_stride_lse,
      .nhead_stride_o = nhead_stride_o,
      .nhead_stride_q_descale = 0,
      .nhead_stride_k_descale = 0,
      .nhead_stride_v_descale = 0,
      .batch_stride_q = batch_stride_q,
      .batch_stride_k = batch_stride_k,
      .batch_stride_v = batch_stride_v,
      .batch_stride_bias = 0,
      .batch_stride_randval = batch_stride_randval,
      .batch_stride_lse = batch_stride_lse,
      .batch_stride_o = batch_stride_o,
      .batch_stride_q_descale = 0,
      .batch_stride_k_descale = 0,
      .batch_stride_v_descale = 0,

      .window_size_left = mask.left,
      .window_size_right = mask.right,
      .sink_size = 0,
      .mask_type = static_cast<ck_tile::index_t>(mask.type),
      .min_seqlen_q = min_seqlen_q,

      .p_drop = dropout_p,
      .s_randval = return_dropout_randval,
      .drop_seed_offset = std::make_pair(rng_ptrs.seed, rng_ptrs.offset),

      .block_scale_size_q = 0,
      .block_scale_size_kv = 0
  };

  auto stream_config = mha_utils::create_stream_config(stream);
  float elapsed = aiter::mha_fwd(args, stream_config);

  if (elapsed < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "aiter::mha_fwd failed - unsupported configuration");
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    aiter_mha_fwd, jax_aiter::aiter_mha_fwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: 4D [b,s,h,d] or 3D [total,h,d]
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_kv (optional)
        .Arg<ffi::AnyBuffer>() // out_provided (optional)
        .Arg<ffi::AnyBuffer>() // bias (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // o
        .Ret<ffi::AnyBuffer>() // lse
        .Ret<ffi::AnyBuffer>() // p (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval")
        .Attr<bool>("use_asm_v3")
        .Attr<int>("how_v3_bf16_cvt")
        .Attr<int>("max_seqlen_q_attr")
        .Attr<int>("max_seqlen_k_attr")
        .Attr<int>("min_seqlen_q")
        .Attr<float>("logits_soft_cap")
        .Attr<bool>("zero_tensors"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
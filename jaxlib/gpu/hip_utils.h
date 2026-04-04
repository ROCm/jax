// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "logging.h"
#include <hip/hip_runtime.h>

namespace jax_aiter {

inline void hipCheck(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    JA_LOG("HIP error at %s:%d: %s", file, line, hipGetErrorString(err));
    std::abort();
  }
}
#define HIP_CHECK(expr) ::jax_aiter::hipCheck((expr), __FILE__, __LINE__)

// Return current device index (defensive wrapper).
inline int current_device() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  return dev;
}

// Prefer this when you have a device (or managed) pointer.
// Tries to infer the device owning the pointer; falls back to current device.
inline int device_from_ptr(const void *ptr) {
  if (ptr == nullptr)
    return current_device();

  hipPointerAttribute_t attr{};
  hipError_t st = hipPointerGetAttributes(&attr, ptr);
  if (st == hipSuccess) {
    // Most ROCm versions expose 'attr.device'. If it's valid, use it.
    if (attr.device >= 0) {
      return attr.device;
    }
  } else {
    // Not a device/managed pointer (or unknown); keep going with fallback.
    JA_LOG("hipPointerGetAttributes failed in device_from_ptr: %s",
           hipGetErrorString(st));
  }
  return current_device();
}
} // namespace jax_aiter

// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "logging.h"
#include "xla/ffi/api/ffi.h"

namespace jax_aiter {

inline ffi::Error require_env(const char *name, std::string *out) {
  const char *v = std::getenv(name);
  if (!v) {
    LOG("Environment variable %s is not set", name);
    return ffi::Error::InvalidArgument("missing env var");
  }
  *out = v;
  return ffi::Error::Success();
}

} // namespace jax_aiter

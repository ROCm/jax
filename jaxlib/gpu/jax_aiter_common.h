// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// (Ruturaj4): lightweight than something like iostream, just for fprintf
#include <cstdio>

namespace jax_aiter {

#ifdef DEBUG_LOGGING
#define LOG(msg, ...)                                                          \
  do {                                                                         \
    fprintf(stderr, "[JAX_AITER] " msg "\n", ##__VA_ARGS__);                   \
  } while (0)
#else
#define LOG(msg, ...)                                                          \
  do {                                                                         \
  } while (0)
#endif

} // namespace jax_aiter
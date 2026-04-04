// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifdef JAX_AITER_DEBUG
#include <cstdio>
#define JA_LOG(fmt, ...)                                                       \
  fprintf(stderr, "[JAX_AITER_CPP] " fmt "\n", ##__VA_ARGS__)
#else
#define JA_LOG(fmt, ...) ((void)0)
#endif

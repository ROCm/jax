
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Public exports for the `jax_aiter.mha` sub-package.

Uses unified AITER entry point (aiter::mha_fwd / aiter::mha_bwd) for both
batch and variable-length attention. CK vs ASM v3 dispatch handled internally
by AITER.

Public API:
    flash_attn_func: Batch flash attention with custom_vjp
    flash_attn_varlen: Variable-length flash attention with custom_vjp
"""

from .aiter_mha import (
    flash_attn_func,
    flash_attn_varlen,
)

__all__ = [
    "flash_attn_func",
    "flash_attn_varlen",
]
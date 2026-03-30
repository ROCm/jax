# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AITer FFI kernels for ROCm, loaded via the jax-rocm plugin wheel."""

import jax
import numpy as np
from typing import Any
from jaxlib.plugin_support import import_from_plugin

_hip_aiter = import_from_plugin("rocm", "_aiter")
def aiter_registrations() -> dict[str, list[tuple[str, Any, int]]]:
  registrations: dict[str, list[tuple[str, Any, int]]] = {
      "ROCM": [],
  }
  if _hip_aiter:
    registrations["ROCM"].extend(
        (name, value, int(name.endswith("_ffi")))
        for name, value in _hip_aiter.registrations().items()
    )
  return registrations


#if hasattr(gpu_aiter, "registrations"):
for platform, targets in aiter_registrations().items():
  for name, value, api_version in targets:
    jax.ffi.register_ffi_target(
        name, value, platform=platform, api_version=api_version
    )


def neuron_fwd(a, b):
  assert a.dtype == np.float32
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jax.ffi.ffi_call(
      "hip_neuron_fwd_ffi", (out_type, out_type),
      vmap_method="sequential")(a, b, n=n)
  return c, (a, b_plus_1)


def neuron_bwd(res, c_grad):
  a, b_plus_1 = res
  assert c_grad.dtype == np.float32
  assert c_grad.shape == a.shape
  assert a.shape == b_plus_1.shape
  assert c_grad.dtype == a.dtype
  assert a.dtype == b_plus_1.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  return jax.ffi.ffi_call(
      "hip_neuron_bwd_ffi", (out_type, out_type),
      vmap_method="sequential")(c_grad, a, b_plus_1, n=n)


@jax.custom_vjp
def neuron(a, b):
  c, _ = neuron_fwd(a, b)
  return c


neuron.defvjp(neuron_fwd, neuron_bwd)

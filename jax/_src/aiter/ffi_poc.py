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
"""AITer FFI kernels for ROCm, loaded via the jax-rocm plugin wheel.

The _aiter.so nanobind extension is built in jaxlib/rocm/ and ships inside
the jax-rocm7-plugin wheel.  This module loads it via import_from_plugin
(the same mechanism used by gpu_rnn.py / gpu_solver.py) and registers the
custom-call targets with XLA.
"""

import jax
import jax.numpy as jnp
from jaxlib import xla_client
from jaxlib.plugin_support import import_from_plugin
import numpy as np

_hip_aiter = import_from_plugin("rocm", "_aiter")

if _hip_aiter:
  for _name, _value in _hip_aiter.registrations().items():
    api_version = 1 if _name.endswith("_ffi") else 0
    xla_client.register_custom_call_target(
        _name, _value, platform="ROCM", api_version=api_version)
    print(f"[aiter] registered: {_name} (api_version={api_version})")
else:
  print("[aiter] WARNING: _aiter module not loaded (_hip_aiter is None). "
        "FFI handlers will NOT be registered.")


def neuron_fwd(a, b):
  assert a.dtype == jnp.float32
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
  assert c_grad.dtype == jnp.float32
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

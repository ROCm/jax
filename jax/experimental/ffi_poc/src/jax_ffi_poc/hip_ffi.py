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
"""An end-to-end example demonstrating the use of the JAX FFI with CUDA.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os
import ctypes
import jax
import jax.numpy as jnp
import numpy as np


# Load the shared library with the FFI target definitions
# The library is installed alongside this module by scikit-build-core
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "lib_hip_ffi_poc.so")
library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

#library.neuron_fwd.restype = ctypes.c_void_p
#library.neuron_bwd.restype = ctypes.c_void_p


jax.ffi.register_ffi_target("neuron_fwd_tn", jax.ffi.pycapsule(library.neuron_fwd), platform="ROCM")
jax.ffi.register_ffi_target("neuron_bwd_tn", jax.ffi.pycapsule(library.neuron_bwd), platform="ROCM")


def neuron_fwd(a, b):
  assert a.dtype == jnp.float32
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jax.ffi.ffi_call("neuron_fwd_tn", (out_type, out_type), vmap_method="sequential")(a, b, n=n)
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
  return jax.ffi.ffi_call("neuron_bwd_tn", (out_type, out_type), vmap_method="sequential")(c_grad, a, b_plus_1,
                          n=n)


@jax.custom_vjp
def neuron(a, b):
  c, _ = neuron_fwd(a, b)
  return c


neuron.defvjp(neuron_fwd, neuron_bwd)

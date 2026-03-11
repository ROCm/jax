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
"""An end-to-end example demonstrating the use of the JAX FFI with HIP.

The specifics of the kernels are not very important, but the general structure,
and packaging of the extension are useful for testing.
"""

import os
import ctypes

import numpy as np

import jax
import jax.numpy as jnp

# Load the shared library with the FFI target definitions
SHARED_LIBRARY = os.path.join(os.path.dirname(__file__), "lib_hip_examples.so")
lib = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

fwd_addr = ctypes.cast(lib.FooFwd, ctypes.c_void_p).value
bwd_addr = ctypes.cast(lib.FooBwd, ctypes.c_void_p).value

print("FooFwd addr:", hex(fwd_addr))
print("FooBwd addr:", hex(bwd_addr))

jax.ffi.register_ffi_target("foo_fwd_t", jax.ffi.pycapsule(lib.FooFwd),
                            platform="ROCM")
jax.ffi.register_ffi_target("foo_bwd_t", jax.ffi.pycapsule(lib.FooBwd),
                            platform="ROCM")

print(f"lib.FooFwd = {lib.FooFwd}")
print("addr:", ctypes.cast(lib.FooFwd, ctypes.c_void_p).value)
print(f"jax.ffi.pycapsule(lib.FooFwd) = {jax.ffi.pycapsule(fwd_addr)}")



def foo_fwd(a, b):
  assert a.dtype == jnp.float32
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jax.ffi.ffi_call("foo_fwd_t", (out_type, out_type))(a, b, n=n)
  return c, (a, b_plus_1)


def foo_bwd(res, c_grad):
  a, b_plus_1 = res
  assert c_grad.dtype == jnp.float32
  assert c_grad.shape == a.shape
  assert a.shape == b_plus_1.shape
  assert c_grad.dtype == a.dtype
  assert a.dtype == b_plus_1.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  return jax.ffi.ffi_call("foo_bwd_t", (out_type, out_type))(c_grad, a, b_plus_1,
                          n=n)


@jax.custom_vjp
def foo(a, b):
  c, _ = foo_fwd(a, b)
  return c


foo.defvjp(foo_fwd, foo_bwd)

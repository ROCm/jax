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


import jax
import jax.numpy as jnp
import pdb

import numpy as np

def allclose(a, b):
  print("a=", a)
  print("b=", b)
  np.allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False)

 
from jax_ffi_example import hip_examples  # pylint: disable=g-import-not-at-top

'''
class hipE2eTests(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["rocm"]):
      self.skipTest("Unsupported platform")

    # Import here to avoid trying to load the library when it's not built.
    from jax_ffi_example import hip_examples  # pylint: disable=g-import-not-at-top

    self.foo = hip_examples.foo

  def test_fwd_interpretable(self):
    shape = (2, 3)
    a = 2.0 * jnp.ones(shape, dtype=jnp.float32)
    b = 3.0 * jnp.ones(shape, dtype=jnp.float32)
    observed = jax.jit(self.foo)(a, b)
    expected = 2.0 * (3.0 + 1.0)
    self.assertArraysEqual(observed, jnp.float32(expected))

  def test_bwd_interpretable(self):
    shape = (2, 3)
    a = 2.0 * jnp.ones(shape, dtype=jnp.float32)
    b = 3.0 * jnp.ones(shape, dtype=jnp.float32)

    def loss(a, b):
      return jnp.sum(self.foo(a, b))

    da_observed, db_observed = jax.jit(jax.grad(loss, argnums=(0, 1)))(a, b)
    da_expected = b + 1
    db_expected = a
    self.assertArraysEqual(da_observed, da_expected)
    self.assertArraysEqual(db_observed, db_expected)

  def test_fwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape, dtype=jnp.float32)
    b = jax.random.normal(key=bkey, shape=shape, dtype=jnp.float32)
    observed = jax.jit(self.foo)(a, b)
    expected = a * (b + 1)
    self.assertAllClose(observed, expected)

  def test_bwd_random(self):
    shape = (2, 3)
    akey, bkey = jax.random.split(jax.random.key(0))
    a = jax.random.normal(key=akey, shape=shape, dtype=jnp.float32)
    b = jax.random.normal(key=bkey, shape=shape, dtype=jnp.float32)
    jtu.check_grads(f=jax.jit(self.foo), args=(a, b), order=1, modes=("rev",))


#if __name__ == "__main__":
#  absltest.main(testLoader=jtu.JaxTestLoader())

'''
def test_fwd_interpretable2():
  print("test_fwd_interpretable2")
  shape = (2, 3)
  a = 2.0 * jnp.ones(shape, dtype=jnp.float32)
  b = 3.0 * jnp.ones(shape, dtype=jnp.float32)
  observed = jax.jit(hip_examples.foo(a, b))
  expected = 2.0 * (3.0 + 1.0)
  allclose(observed, expected)
    

#test_fwd_interpretable2

def test_fwd_random2():
  print("test_fwd_random2")
  shape = (2, 3)
  akey, bkey = jax.random.split(jax.random.key(0))
  a = jax.random.normal(key=akey, shape=shape, dtype=jnp.float32)
  b = jax.random.normal(key=bkey, shape=shape, dtype=jnp.float32)
  #pdb.set_trace()
  observed = hip_examples.foo(a, b)
  expected = a * (b + 1)
  allclose(observed, expected)

test_fwd_random2()

'''
print(f"\n[5] Test FFI Call")
try:
    a = jnp.ones((4,), dtype=jnp.float32)
    b = jnp.ones((4,), dtype=jnp.float32) * 2
    c = jnp.ones((4,), dtype=jnp.float32)
    
    import numpy as np
    n = np.prod(a.shape).astype(np.uint64)
    out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
    
    @jax.jit
    def test_fn(a, b, c):

        pdb.set_trace()
        c, b_plus_1 = jax.ffi.ffi_call(
            "foo-fwd",
            (out_type, out_type),
            vmap_method="sequential",
        )(a, b, c, n=n)
        return c
    
    result = test_fn(a, b, c)
    expected = a * (b + 1)
    
    print(f"    Input a:    {a}")
    print(f"    Input b:    {b}")
    print(f"    Result:     {result}")
    print(f"    Expected:   {expected}")
    print(f"    Match:      {'✓ PASS' if jnp.allclose(result, expected) else '✗ FAIL'}")
    
except Exception as e:
    print(f"    ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
'''

# Copyright 2025 The JAX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transform inference tests for the Mosaic GPU MLIR dialect."""

# pylint: disable=g-complex-comprehension

from absl.testing import parameterized
import jax
from jax import numpy as jnp
from jax._src import config
from jax._src import test_util as jtu
from jax._src.interpreters import mlir as mlir_interpreter
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith
from jax._src.lib.mlir.dialects import func
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import inference_utils
import numpy as np


config.parse_flags_with_absl()


def _make_ir_context():
  context = ir.Context()
  context.append_dialect_registry(mlir_interpreter.upstream_dialects)
  context.load_all_available_dialects()
  mgpu.dialect.register_dialect(context)
  return context


class TransformInferenceTest(parameterized.TestCase):

  def setUp(self):
    if jax.version._version != jax.lib.__version__:
      raise self.skipTest("Test requires matching jax and jaxlib versions")
    super().setUp()
    self.enter_context(_make_ir_context())
    self.enter_context(ir.Location.unknown())
    self.module = ir.Module.create()

  @parameterized.parameters(
      (swizzle, dtype)
      for swizzle in mgpu.dialect.SwizzlingMode
      for dtype in [jnp.bfloat16, jnp.float32]
  )
  def test_infer_transforms_for_wgmma_op(self, swizzle, dtype):
    swizzle_elems = swizzle // np.dtype(dtype).itemsize
    m = 64
    # Note: `group_m` and `group_k` should be coprime with 2 for the test to be
    # correct. Otherwise, we may infer larger swizzles than the test intends to
    # check.
    group_m, group_k = 3, 3
    lhs_shape = (group_m * m, group_k * swizzle_elems)
    rhs_shape = (group_k * swizzle_elems, group_k * swizzle_elems)
    out_shape = (group_m * m, group_k * swizzle_elems)
    wgmma_op = None

    def body(accumulator, lhs, rhs):
      nonlocal wgmma_op
      wgmma_op = mgpu.dialect.WGMMAOp(accumulator, lhs, rhs)

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      elt_ty = mgpu.utils.dtype_to_ir_type(dtype)
      lhs_ty = ir.MemRefType.get(lhs_shape, elt_ty, memory_space=smem)
      rhs_ty = ir.MemRefType.get(rhs_shape, elt_ty, memory_space=smem)
      acc_ty = ir.VectorType.get(out_shape, elt_ty)
      func.FuncOp.from_py_func(acc_ty, lhs_ty, rhs_ty)(body)

    mgpu.infer_transforms(self.module)

    arg_transforms = ir.ArrayAttr.get([
        mgpu.dialect.TileTransformAttr.get((8, swizzle_elems)),
        mgpu.dialect.SwizzleTransformAttr.get(int(swizzle)),
    ])

    self.assertSequenceEqual(
        inference_utils.in_transforms(wgmma_op),
        [arg_transforms, arg_transforms],
    )
    self.assertEmpty(inference_utils.out_transforms(wgmma_op))

  def test_infer_transforms_for_async_load_derives_from_destination(self):
    async_load_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(gmem_ref, smem_ref, barrier):
      nonlocal async_load_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_load_op = mgpu.dialect.AsyncLoadOp(
          source=gmem_ref,
          destination=smem_ref,
          barrier=barrier,
          indices=[zero, zero],
          slice_lengths=shape,
          collective=ir.ArrayAttr.get([]),
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      barrier_ty = ir.Type.parse("!mosaic_gpu.barrier")
      f = func.FuncOp.from_py_func(gmem_ty, smem_ty, barrier_ty)(body).func_op

    transforms = ir.ArrayAttr.get(
        [mgpu.dialect.TransposeTransformAttr.get((1, 0))]
    )
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_load_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_load_op))

  def test_infer_transforms_for_async_store_op_derives_from_source(self):
    async_store_op = None
    shape = (64, 64)
    elt_ty = ir.BF16Type.get()

    def body(gmem_ref, smem_ref):
      nonlocal async_store_op
      zero = arith.constant(ir.IntegerType.get_signless(32), 0)
      async_store_op = mgpu.dialect.AsyncStoreOp(
          source=smem_ref,
          destination=gmem_ref,
          indices=[zero, zero],
          slice_lengths=shape,
      )

    with ir.InsertionPoint(self.module.body):
      smem = ir.Attribute.parse("#gpu.address_space<workgroup>")
      gmem_ty = ir.MemRefType.get(shape, elt_ty)
      smem_ty = ir.MemRefType.get(shape, elt_ty, memory_space=smem)
      f = func.FuncOp.from_py_func(gmem_ty, smem_ty)(body).func_op

    transforms = ir.ArrayAttr.get(
        [mgpu.dialect.TransposeTransformAttr.get((1, 0))]
    )
    f.attributes["in_transforms"] = ir.ArrayAttr.get([transforms])

    mgpu.infer_transforms(self.module)

    self.assertSequenceEqual(
        inference_utils.in_transforms(async_store_op), [transforms]
    )
    self.assertEmpty(inference_utils.out_transforms(async_store_op))


if __name__ == "__main__":
  parameterized.absltest.main(testLoader=jtu.JaxTestLoader())

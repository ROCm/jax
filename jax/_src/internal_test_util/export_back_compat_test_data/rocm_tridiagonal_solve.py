# Copyright 2025 The JAX Authors.
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

# ruff: noqa

import datetime
import numpy as np

array = np.array
float32 = np.float32

data_2026_01_05 = {}

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_01_05["f32"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsparse_gtsv2_ffi'],
    serialized_date=datetime.date(2026, 1, 5),
    inputs=(array([ 0., 29., 81., 79.,  2., 70., 21., 64.,  1., 74.], dtype=float32), array([71., 40., 67., 27., 66., 53., 89., 13., 78., 71.], dtype=float32), array([34., 55.,  3., 31.,  7., 89., 16., 29., 98.,  0.], dtype=float32), array([[61.],
       [42.],
       [53.],
       [36.],
       [43.],
       [89.],
       [52.],
       [58.],
       [46.],
       [88.]], dtype=float32)),
    expected_outputs=(array([[ 0.5840358 ],
       [ 0.5745135 ],
       [ 0.03786218],
       [ 1.3092135 ],
       [-0.07547983],
       [ 6.4804626 ],
       [-2.7997856 ],
       [10.318197  ],
       [ 3.5534375 ],
       [-2.464146  ]], dtype=float32),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10xf32> loc("dl"), %arg1: tensor<10xf32> loc("d"), %arg2: tensor<10xf32> loc("du"), %arg3: tensor<10x1xf32> loc("b")) -> (tensor<10x1xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @hipsparse_gtsv2_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i], [j], [k], [l, m])->([n, o]) {i=10, j=10, k=10, l=10, m=1, n=10, o=1}, custom>} : (tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10x1xf32>) -> tensor<10x1xf32> loc(#loc7)
    return %0 : tensor<10x1xf32> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":794:4)
#loc6 = loc("jit(tridiagonal_solve)"(#loc5))
#loc7 = loc("tridiagonal_solve"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa7}\x13\x015\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03/\x0b/O\x1b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0f\x13\x0f\x05\x1bW\x0f\x0f\x0f\x0f\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x0f\x13\x17\x07\x07#\x13\x13\x02\xfa\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x1d\x1b\x01\x05\x1b\x1d\x1f\x01\x05\x1d\x03\x07#5%K'c\x05\x1f\x05!\x05#\x1d+-\x05%\x1d/1\x05'\x173j\x0c\t\x05)\r\x01\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\t5555#\r\x03\x03A\r\x03CE\x1d+\x1d-\x1d/\x1d1\r\x03MO\x1d3\x1d5\x0b\x03\x1d7\x1d9\x03\x01\x05\x01\x03\t7779\x03\x03_\x15\x01\r\x01\x03\x039\x15\x0f))))\x05)\x05\teimq\x03w\x01\x01\x01\x01\x01\x13\x03g\x11\x03\x01\x13\x03k\x11\x03\x05\x13\x03o\x11\x03\t\x13\x05su\x11\x03\r\x11\x03\x11\x13\x05y{\x11\x03\x15\x11\x03\x19\x01\t\x01\x02\x02)\x03)\t)\x05)\x05\t\t\x13\x11\t\x05\x05\x05\x07\x03\x07)\x03\x05\x0b)\x03\t\x0b\x04c\x05\x01Q\x01\x05\x01\x07\x04Q\x03\x01\x05\x03P\x01\x03\x07\x04=\x03\x0b\x0b\t\x0b\x11\x0b\x15\x0b\x19\x0f\x1d\x00\x05G)!\x05\x03\x07\t\x01\x03\x05\x07\x07\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\x8e\x06;)\x03\x05\x1f\x0f\x0b\x0f!s/%%3)\x05\x07\x05\x07-%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00tridiagonal_solve\x00jit(tridiagonal_solve)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x000\x00\x00hipsparse_gtsv2_ffi\x00\x08'\x07\x05\x1f\x01\x0b;=?GI\x11QSUWY[]a",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste

# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_2026_01_05["f64"] = dict(
    testdata_version=1,
    platform='rocm',
    custom_call_targets=['hipsparse_gtsv2_ffi'],
    serialized_date=datetime.date(2026, 1, 5),
    inputs=(array([ 0., 29., 81., 79.,  2., 70., 21., 64.,  1., 74.]), array([71., 40., 67., 27., 66., 53., 89., 13., 78., 71.]), array([34., 55.,  3., 31.,  7., 89., 16., 29., 98.,  0.]), array([[61.],
       [42.],
       [53.],
       [36.],
       [43.],
       [89.],
       [52.],
       [58.],
       [46.],
       [88.]])),
    expected_outputs=(array([[ 0.5840358128194283 ],
       [ 0.5745134497006055 ],
       [ 0.03786224436749748],
       [ 1.3092134005428773 ],
       [-0.0754799716029022 ],
       [ 6.480464474957971  ],
       [-2.7997867321412278 ],
       [10.318204074153247  ],
       [ 3.553437858380911  ],
       [-2.4641465002843295 ]]),),
    mlir_module_text=r"""
#loc1 = loc("dl")
#loc2 = loc("d")
#loc3 = loc("du")
#loc4 = loc("b")
module @jit_tridiagonal_solve attributes {jax.uses_shape_polymorphism = false, mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<10xf64> loc("dl"), %arg1: tensor<10xf64> loc("d"), %arg2: tensor<10xf64> loc("du"), %arg3: tensor<10x1xf64> loc("b")) -> (tensor<10x1xf64> {jax.result_info = "result"}) {
    %0 = stablehlo.custom_call @hipsparse_gtsv2_ffi(%arg0, %arg1, %arg2, %arg3) {mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<0> : tensor<1xindex>, dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 3, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i], [j], [k], [l, m])->([n, o]) {i=10, j=10, k=10, l=10, m=1, n=10, o=1}, custom>} : (tensor<10xf64>, tensor<10xf64>, tensor<10xf64>, tensor<10x1xf64>) -> tensor<10x1xf64> loc(#loc7)
    return %0 : tensor<10x1xf64> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc = loc(unknown)
#loc5 = loc("/workspace/rocm-jax/jax/tests/export_back_compat_test.py":794:4)
#loc6 = loc("jit(tridiagonal_solve)"(#loc5))
#loc7 = loc("tridiagonal_solve"(#loc6))
""",
    mlir_module_serialized=b"ML\xefR\rStableHLO_v1.12.1\x00\x01\x1b\x07\x01\x05\t\t\x01\x03\x0f\x03\x07\x13\x17\x1b\x03\xa7}\x13\x015\x07\x0f#\x0b\x0f\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x0f\x0b\x0f\x0b#\x0b\x0b\x0b\x0f\x0b\x0f\x0b\x17\x0b\x03/\x0b/O\x1b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x13\x0b\x0b\x0b\x0b\x0b\x0b\x0b\x1b\x0f\x13\x0f\x05\x1bW\x0f\x0f\x0f\x0f\x0f\x0f\x13\x0f\x0f\x13\x0f\x0f\x01\x05\x0b\x0f\x03\x0f\x13\x17\x07\x07#\x13\x13\x02\xfa\x03\x1f\x11\x03\x05\x03\x07\x07\t\x0b\x03\r\x03\x05\x0f\x11\x01\x00\x05\x11\x05\x13\x05\x15\x1d\x13\x01\x05\x17\x1d\x17\x01\x05\x19\x1d\x1b\x01\x05\x1b\x1d\x1f\x01\x05\x1d\x03\x07#5%K'c\x05\x1f\x05!\x05#\x1d+-\x05%\x1d/1\x05'\x173j\x0c\t\x05)\r\x01\x1f\x0f\x11\x00\x00\x00\x00\x00\x00\x00\x00\x1f\x11!\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x03\t5555#\r\x03\x03A\r\x03CE\x1d+\x1d-\x1d/\x1d1\r\x03MO\x1d3\x1d5\x0b\x03\x1d7\x1d9\x03\x01\x05\x01\x03\t7779\x03\x03_\x15\x01\r\x01\x03\x039\x15\x0f))))\x05)\x05\teimq\x03w\x01\x01\x01\x01\x01\x13\x03g\x11\x03\x01\x13\x03k\x11\x03\x05\x13\x03o\x11\x03\t\x13\x05su\x11\x03\r\x11\x03\x11\x13\x05y{\x11\x03\x15\x11\x03\x19\x01\t\x01\x02\x02)\x03)\t)\x05)\x05\t\x0b\x13\x11\t\x05\x05\x05\x07\x03\x07)\x03\x05\x0b)\x03\t\x0b\x04c\x05\x01Q\x01\x05\x01\x07\x04Q\x03\x01\x05\x03P\x01\x03\x07\x04=\x03\x0b\x0b\t\x0b\x11\x0b\x15\x0b\x19\x0f\x1d\x00\x05G)!\x05\x03\x07\t\x01\x03\x05\x07\x07\x04\x01\x03\t\x06\x03\x01\x05\x01\x00\x8e\x06;)\x03\x05\x1f\x0f\x0b\x0f!s/%%3)\x05\x07\x05\x07-%)9\x15\x1f\x11\x0f\t\x0b\x11builtin\x00vhlo\x00sdy\x00module\x00func_v1\x00custom_call_v1\x00return_v1\x00jax.uses_shape_polymorphism\x00mhlo.num_partitions\x00mhlo.num_replicas\x00jit_tridiagonal_solve\x00dl\x00d\x00du\x00b\x00mhlo.backend_config\x00mhlo.frontend_attributes\x00sdy.sharding_rule\x00tridiagonal_solve\x00jit(tridiagonal_solve)\x00/workspace/rocm-jax/jax/tests/export_back_compat_test.py\x00jax.result_info\x00result\x00main\x00public\x00num_batch_dims\x000\x00\x00hipsparse_gtsv2_ffi\x00\x08'\x07\x05\x1f\x01\x0b;=?GI\x11QSUWY[]a",
    xla_call_module_version=10,
    nr_devices=1,
)  # End paste


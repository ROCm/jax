#!/usr/bin/env python3
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

"""
Configuration file containing the list of multi-GPU test files.
This list is shared between run_single_gpu.py and run_multi_gpu.sh
to ensure consistency and avoid duplication.
"""

# Multi-GPU test files that should be excluded from single GPU runs
# but included in multi-GPU test runs
MULTI_GPU_TESTS = {
    "tests/multiprocess_gpu_test.py",
    "tests/debug_info_test.py",
    "tests/checkify_test.py",
    "tests/mosaic/gpu_test.py",
    "tests/random_test.py",
    "tests/jax_jit_test.py",
    "tests/mesh_utils_test.py",
    "tests/pjit_test.py",
    "tests/linalg_sharding_test.py",
    "tests/multi_device_test.py",
    "tests/distributed_test.py",
    "tests/shard_alike_test.py",
    "tests/api_test.py",
    "tests/ragged_collective_test.py",
    "tests/batching_test.py",
    "tests/scaled_matmul_stablehlo_test.py",
    "tests/export_harnesses_multi_platform_test.py",
    "tests/pickle_test.py",
    "tests/roofline_test.py",
    "tests/profiler_test.py",
    "tests/error_check_test.py",
    "tests/debug_nans_test.py",
    "tests/shard_map_test.py",
    "tests/colocated_python_test.py",
    "tests/cudnn_fusion_test.py",
    "tests/compilation_cache_test.py",
    "tests/export_back_compat_test.py",
    "tests/pgle_test.py",
    "tests/ffi_test.py",
    "tests/lax_control_flow_test.py",
    "tests/fused_attention_stablehlo_test.py",
    "tests/layout_test.py",
    "tests/pmap_test.py",
    "tests/aot_test.py",
    "tests/mock_gpu_topology_test.py",
    "tests/ann_test.py",
    "tests/debugging_primitives_test.py",
    "tests/array_test.py",
    "tests/export_test.py",
    "tests/memories_test.py",
    "tests/debugger_test.py",
    "tests/python_callback_test.py",
}

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # Print all multi-GPU tests, one per line (for bash scripts)
        for test in MULTI_GPU_TESTS:
            print(test)
    else:
        print("Multi-GPU tests configuration")
        print(f"Total tests: {len(MULTI_GPU_TESTS)}")
        print("\nTo get the list for bash scripts, run:")
        print("python3 multi_gpu_tests_config.py --list")


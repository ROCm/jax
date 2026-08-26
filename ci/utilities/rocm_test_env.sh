#!/bin/bash
# Copyright 2026 The JAX Authors.
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
#
# The environment the ROCm pytest runs share. Sourced by
# ci/utilities/prepare_rocm_tests.sh, and on its own by anything that needs to
# reproduce the environment of a run without starting one. Requires
# ci/envs/default.env to have been sourced already.

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"

# ==============================================================================
# Number of parallel processes for pytest: 4 test workers per GPU (matches SPX).
# ==============================================================================

export gpu_count=$(rocminfo | egrep -c "Device Type:\s+GPU")
echo "Number of GPUs detected: $gpu_count"

# Fail fast when the runner label promises N GPUs but ROCr/JAX see fewer. This
# catches duplicate ROCR_VISIBLE_DEVICES entries on DPX pods that otherwise skip
# multi-accelerator tests and still report success.
runner_label="${INPUT_RUNNER:-${RUNNER_LABEL:-${GITHUB_RUNNER:-}}}"
expected_gpus=0
case "$runner_label" in
  *gfx950.8*|*8-dpx*) expected_gpus=8 ;;
  *gfx950.4*|*4-dpx*) expected_gpus=4 ;;
  *gfx950.1*|*1-dpx*) expected_gpus=1 ;;
esac

echo "Runner label context: ${runner_label:-unset}"
echo "Expected GPUs from runner label: ${expected_gpus:-unknown}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"

jax_device_count=$("$JAXCI_PYTHON" -c "import jax; print(len(jax.devices()))")
echo "jax.devices() count: $jax_device_count"

if [[ "$gpu_count" -eq 0 ]]; then
  echo "ERROR: rocminfo detected no GPUs" >&2
  exit 1
fi

if [[ "$gpu_count" -ne "$jax_device_count" ]]; then
  echo "ERROR: rocminfo ($gpu_count) != jax.devices() ($jax_device_count)" >&2
  exit 1
fi

if [[ "$expected_gpus" -gt 0 && "$gpu_count" -ne "$expected_gpus" ]]; then
  echo "ERROR: runner expects $expected_gpus GPU(s), found $gpu_count" >&2
  echo "Check /etc/podinfo/gha-gpu-isolation-settings for duplicate ROCR indices." >&2
  exit 1
fi

workers_per_gpu="${ROCM_PYTEST_WORKERS_PER_GPU:-4}"
export num_processes=$((gpu_count * workers_per_gpu))
echo "Workers per GPU: $workers_per_gpu"
echo "Number of processes to run: $num_processes"

export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=address
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

# Deselected by both subsets, listed here so the two runs cannot drift apart.
# Expanded unquoted by the callers, which is why no entry may contain a space.
export rocm_pytest_deselect="\
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric"

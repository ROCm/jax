#!/bin/bash
# Copyright 2024 The JAX Authors.
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
# Runs Pytest ROCm tests. Requires the jaxlib and ROCm plugin wheels to be
# present inside $JAXCI_OUTPUT_DIR (../dist)
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Source default JAXCI environment variables.
source ci/envs/default.env

# Install jaxlib and ROCm plugin wheels inside the $JAXCI_OUTPUT_DIR directory
echo "Installing wheels locally..."
source ./ci/utilities/install_wheels_locally.sh

# Print all the installed packages
echo "Installed packages:"
"$JAXCI_PYTHON" -m uv pip freeze

"$JAXCI_PYTHON" -c "import jax; print(jax.default_backend()); print(jax.devices()); print(len(jax.devices()))"

# TheRock (pip) ROCm images are intentionally /opt/rocm-less, so ROCm CLI tools
# (rocminfo, rocm-smi, ...) live in the SDK bin dir rather than on PATH. Put
# them on PATH via rocm-sdk. apt-ROCm images lack rocm-sdk and already have
# these tools on PATH, so the gate leaves them untouched.
if command -v rocm-sdk >/dev/null 2>&1; then
  sdk_bin="$(rocm-sdk path --bin)"
  if [[ -n "$sdk_bin" ]]; then
    export PATH="$sdk_bin:$PATH"
  fi
fi

rocm-smi

# ==============================================================================
# Set up the generic test environment variables
# ==============================================================================
export PY_COLORS=1
export JAX_SKIP_SLOW_TESTS=true
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="$JAXCI_ENABLE_X64"

# ==============================================================================
# Calculate the optimal number of parallel processes for pytest
# This will be the minimum of: GPU capacity, CPU core count, and a system RAM limit.
# ==============================================================================

export gpu_count=$(rocminfo | egrep -c "Device Type:\s+GPU")
echo "Number of GPUs detected: $gpu_count"

# Query GPU 0 memory using rocm-smi
export memory_per_gpu_mib=$(rocm-smi -d 0 --showmeminfo vram | grep -i "vram total" | awk '{print int($NF/1024/1024)}' | head -1)
echo "Reported memory per GPU: $memory_per_gpu_mib MiB"

# Convert effective memory from MiB to GiB.
export memory_per_gpu_gib=$((memory_per_gpu_mib / 1024))
echo "Effective memory per GPU: $memory_per_gpu_gib GiB"

# Allow 2 GiB of GPU RAM per test.
export max_tests_per_gpu=$((memory_per_gpu_gib / 2))
echo "Max tests per GPU (assuming 2GiB/test): $max_tests_per_gpu"

export num_processes=$((gpu_count * max_tests_per_gpu))
echo "Initial number of processes based on GPU capacity: $num_processes"

export num_cpu_cores=$(nproc)
echo "Number of CPU cores available: $num_cpu_cores"

# Reads total memory from /proc/meminfo (in KiB) and converts to GiB.
export total_ram_gib=$(awk '/MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo)
echo "Total system RAM: $total_ram_gib GiB"

# Set a safety limit for system RAM usage, e.g., 1/6th of total.
export host_memory_limit=$((total_ram_gib / 6))
echo "Host memory process limit (1/6th of total RAM): $host_memory_limit"

if [[ $num_cpu_cores -lt $num_processes ]]; then
  num_processes=$num_cpu_cores
  echo "Adjusting num_processes to match CPU core count: $num_processes"
fi

if [[ $host_memory_limit -lt $num_processes ]]; then
  num_processes=$host_memory_limit
  echo "Adjusting num_processes to match host memory limit: $num_processes"
fi

if [[ 16 -lt $num_processes ]]; then
  num_processes=16
  echo "Reducing num_processes to $num_processes"
fi

echo "Final number of processes to run: $num_processes"

export JAX_ENABLE_ROCM_XDIST="$gpu_count"
export XLA_PYTHON_CLIENT_ALLOCATOR=address
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

# ==============================================================================
# FLAKE-HUNT LOGGING v2 (branch magaonka/pytest) -- LEAN & BOUNDED.
# v1 backfired: TENSILE_DB=0x6 + AMD_LOG_LEVEL=3 + pytest -rA made a single test's
# captured output exceed 2 GB, overflowing xdist/execnet's 32-bit message length
# (struct.pack "!bii") -> worker channel crash -> INTERNALERROR (run 30402766382).
#
# v2 keeps ONLY bounded logging (no per-call / per-candidate firehose):
#   - pytest -v --tb=long (NOT -rA): verbose names + FULL traceback on FAILURES
#     only (wrong-value assertions show mismatched elements/values). Passed-test
#     output is NOT shipped, so per-test messages stay tiny.
#   - AMD_LOG_LEVEL=1 : HIP ERRORS/faults only (tiny) -> catches crash-type flakes.
#   - HIPBLASLT_LOG_MASK=1 : hipBLASLt ERRORS only (tiny).
#   - TF_CPP_MIN_LOG_LEVEL=0 : XLA default INFO (as upstream), no vmodule firehose.
# Overridable from the workflow env if a targeted deeper run is wanted later.
# ==============================================================================
export AMD_LOG_LEVEL="${AMD_LOG_LEVEL:-1}"
export HIPBLASLT_LOG_MASK="${HIPBLASLT_LOG_MASK:-1}"
export TF_CPP_MIN_LOG_LEVEL=0

# v3 additions, all failure-triggered so passing tests are untouched:
#   - conftest flake forensics: index structure of the mismatching elements
#     (tile/wavefront alignment, constant shift) + which tests were co-resident
#     on the GPU during the failure. Grep the job log for >>>FLAKE-FORENSICS<<<.
#   - post-suite rerun of whatever failed, to separate a transient race from a
#     deterministic bug (see below).
export JAX_FLAKE_FORENSICS=1
export JAX_FLAKE_FORENSICS_DIR="logs/forensics"

# ==============================================================================
# Run tests
# ==============================================================================

# Disable core dumps just in case
ulimit -c 0

echo "Running ROCm tests..."
export NPROC=32
LOGS_DIR="logs"
mkdir -p "${LOGS_DIR}"
mkdir -p test-artifacts

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

# Run single-accelerator tests in parallel
JAX_FLAKE_FORENSICS_TAG=single \
"$JAXCI_PYTHON" -m pytest -n $num_processes -v --tb=long \
--json-report --json-report-file=${LOGS_DIR}/pytest_results_single.json \
--junitxml=test-artifacts/junit-single.xml \
-m "not multiaccelerator" \
--deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
--deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
--deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric \
tests

first_cmd_retval=$?

if [[ $gpu_count -gt 1 ]]; then
  # Run multi-accelerator tests across all GPUs without xdist.
  unset JAX_ENABLE_ROCM_XDIST

  JAX_FLAKE_FORENSICS_TAG=multi \
  "$JAXCI_PYTHON" -m pytest -v --tb=long \
    --json-report --json-report-file=${LOGS_DIR}/pytest_results_multi.json \
    --junitxml=test-artifacts/junit-multi.xml \
    -m "multiaccelerator" \
    --deselect=tests/multi_device_test.py::MultiDeviceTest::test_computation_follows_data \
    --deselect=tests/multiprocess_gpu_test.py::MultiProcessGpuTest::test_distributed_jax_visible_devices \
    --deselect=tests/compilation_cache_test.py::CompilationCacheTest::test_task_using_cache_metric \
    tests

  second_cmd_retval=$?
else
  echo "Skipping multi-accelerator tests (only $gpu_count GPU detected)"
  second_cmd_retval=0
fi

# ==============================================================================
# FLAKE-HUNT: reproduce whatever failed, on the now-idle GPU.
#
# The whole SPX wrong-value family has never reproduced on an idle machine, but
# that has only ever been checked by hand, after the fact. Re-running the exact
# failed node ids here, sequentially (no xdist, no contention), three times,
# classifies every failure automatically as TRANSIENT (race, needs contention)
# or DETERMINISTIC (a real bug) while we still have the machine.
#
# Deliberately does NOT affect the exit code: the main suite stays the clean
# data point, and the classification is read from the summary banner instead.
# ==============================================================================
rerun_transient=0
rerun_deterministic=0
rerun_skipped=0
# Hard caps so a job full of genuine failures can never approach the 120 min
# workflow timeout: at most 12 node ids, 3 attempts each, 10 min per attempt,
# and a 20 min wall-clock budget for the whole rerun phase.
rerun_deadline=$(( $(date +%s) + 20 * 60 ))

mapfile -t failed_nodes < <("$JAXCI_PYTHON" - <<'PYEOF'
import glob, json, sys
seen = []
for path in glob.glob("logs/pytest_results_*.json"):
  try:
    with open(path) as f:
      data = json.load(f)
  except Exception:
    continue
  for test in data.get("tests", []):
    if test.get("outcome") == "failed":
      seen.append(test["nodeid"])
sys.stdout.write("".join(f"{nodeid}\n" for nodeid in dict.fromkeys(seen)))
PYEOF
)

if [[ ${#failed_nodes[@]} -gt 0 ]]; then
  echo "=== FLAKE-HUNT: reproducing ${#failed_nodes[@]} failed test(s) on idle GPU ==="
  node_index=0
  for node in "${failed_nodes[@]}"; do
    node_index=$((node_index + 1))
    if [[ $node_index -gt 12 || $(date +%s) -ge $rerun_deadline ]]; then
      rerun_skipped=$((rerun_skipped + 1))
      continue
    fi
    reproduced=0
    for attempt in 1 2 3; do
      if timeout 600 "$JAXCI_PYTHON" -m pytest -v --tb=line -p no:cacheprovider \
          "$node" > "${LOGS_DIR}/rerun_attempt.log" 2>&1; then
        echo "FLAKE-HUNT rerun ${attempt}/3 PASSED: $node"
      else
        reproduced=$((reproduced + 1))
        echo "FLAKE-HUNT rerun ${attempt}/3 FAILED: $node"
        tail -n 20 "${LOGS_DIR}/rerun_attempt.log"
      fi
      cat "${LOGS_DIR}/rerun_attempt.log" >> "${LOGS_DIR}/rerun_failed_nodes.log"
    done
    if [[ $reproduced -eq 0 ]]; then
      rerun_transient=$((rerun_transient + 1))
      echo ">>>FLAKE-FORENSICS<<< TRANSIENT (0/3 reproduced idle+sequential): $node"
    else
      rerun_deterministic=$((rerun_deterministic + 1))
      echo ">>>FLAKE-FORENSICS<<< DETERMINISTIC (${reproduced}/3 reproduced idle+sequential): $node"
    fi
  done
fi

echo ">>>FLAKE-FORENSICS<<< SUMMARY failed=${#failed_nodes[@]}" \
     "transient=${rerun_transient} deterministic=${rerun_deterministic}" \
     "not_rerun=${rerun_skipped}" \
     "single_rc=${first_cmd_retval} multi_rc=${second_cmd_retval}"

# Exit with failure if either command fails.
if [[ $first_cmd_retval -ne 0 ]]; then
  exit $first_cmd_retval
elif [[ $second_cmd_retval -ne 0 ]]; then
  exit $second_cmd_retval
else
  exit 0
fi

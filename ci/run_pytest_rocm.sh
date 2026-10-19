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
# Runs the single-accelerator tests, then hands the multi-accelerator ones to
# ci/run_pytest_rocm_multi.sh when the host has the GPUs for them.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

source ./ci/utilities/prepare_rocm_tests.sh

# Debug branch only: log per-thread CPU and native stacks if the suite stalls.
# Yama only lets ancestors ptrace by default, so test processes opt in.
"$JAXCI_PYTHON" -m pip install py-spy || true
(apt-get update -qq && apt-get install -y -qq gdb) >/dev/null 2>&1 || echo "gdb install failed"
PYSPY="$("$JAXCI_PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')/py-spy"
hang_debug_dir=$(mktemp -d)
printf 'import ctypes\ntry:\n    ctypes.CDLL(None).prctl(0x59616d61, ctypes.c_ulong(-1), 0, 0, 0)\nexcept Exception:\n    pass\n' \
  > "$hang_debug_dir/sitecustomize.py"
PYTHONPATH="$hang_debug_dir${PYTHONPATH:+:$PYTHONPATH}"
bash ./ci/rocm_hang_watchdog.sh &
watchdog_pid=$!
trap 'kill "$watchdog_pid" 2>/dev/null || true' EXIT

# Swap branch only: run librocprofiler-sdk built from the 10.2.0a20261017 nightly's
# rocm-systems pin plus rocm-systems PR 13720. It is only ABI-matched to that nightly.
cat rprof13720/BUILD_INFO
rprof_nightly_md5=025c95e8dcdc370182932ec95a79b3a7
rprof_swapped=0
for rprof_lib in $(find /usr/local/lib /opt -path '*_rocm_sdk_core/lib/librocprofiler-sdk.so.1' 2>/dev/null); do
  rprof_old_md5=$(md5sum < "$rprof_lib" | cut -d' ' -f1)
  if [[ $rprof_old_md5 != "$rprof_nightly_md5" ]]; then
    echo "$rprof_lib is not the 10.2.0a20261017 build"
    exit 1
  fi
  cp rprof13720/librocprofiler-sdk.so.1 "$rprof_lib.swap"
  mv "$rprof_lib.swap" "$rprof_lib"
  if ldd "$rprof_lib" | grep 'not found'; then
    exit 1
  fi
  echo "RPROF13720_SWAP $rprof_lib before=$rprof_old_md5 after=$(md5sum < "$rprof_lib" | cut -d' ' -f1)"
  rprof_swapped=$((rprof_swapped + 1))
done
if [[ $rprof_swapped -eq 0 ]]; then
  echo "no _rocm_sdk_core librocprofiler-sdk.so.1 found to swap"
  exit 1
fi

# ==============================================================================
# Run tests
# ==============================================================================

echo "Running ROCm tests..."

# Don't abort the script if one command fails to ensure we run both test
# commands below.
set +e

# Run single-accelerator tests in parallel
"$JAXCI_PYTHON" -m pytest -n $num_processes --tb=short \
-o faulthandler_timeout=1200 \
--json-report --json-report-file=${LOGS_DIR}/pytest_results_single.json \
--junitxml=test-artifacts/junit-single.xml \
-m "not multiaccelerator" \
$rocm_pytest_deselect \
tests

first_cmd_retval=$?

if [[ $gpu_count -gt 1 ]]; then
  ./ci/run_pytest_rocm_multi.sh
  second_cmd_retval=$?
else
  echo "Skipping multi-accelerator tests (only $gpu_count GPU detected)"
  second_cmd_retval=0
fi

# Exit with failure if either command fails.
if [[ $first_cmd_retval -ne 0 ]]; then
  exit $first_cmd_retval
elif [[ $second_cmd_retval -ne 0 ]]; then
  exit $second_cmd_retval
else
  exit 0
fi

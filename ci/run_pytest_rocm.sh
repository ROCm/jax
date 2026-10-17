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

# Swap branch only: run these libraries from 10.2.0a20261009 on this image's newer ROCm.
swap_re='_rocm_sdk_core/lib/(libamdhip64|libhiprtc|libhiprtc-builtins)\.so\.7'
core_lib=$(dirname "$(find /usr/local/lib -path '*_rocm_sdk_core/lib/libamdhip64.so.7' | head -1)")
curl -fsSL -o "$hang_debug_dir/old.whl" \
  https://nightly.repo.amd.com/rocm/whl-next/rocm-sdk-core/rocm_sdk_core-10.2.0a20261009-py3-none-linux_x86_64.whl
"$JAXCI_PYTHON" - "$hang_debug_dir/old.whl" "$core_lib" "$swap_re" <<'EOF'
import hashlib, os, re, sys, zipfile
whl, dest, pattern = sys.argv[1:]
with zipfile.ZipFile(whl) as z:
    for name in z.namelist():
        if re.fullmatch(pattern, name):
            target = os.path.join(dest, os.path.basename(name))
            before = hashlib.md5(open(target, "rb").read()).hexdigest()
            data = z.read(name)
            with open(target + ".swap", "wb") as f:
                f.write(data)
            os.replace(target + ".swap", target)
            print("swapped", target, before, "->", hashlib.md5(data).hexdigest())
EOF

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

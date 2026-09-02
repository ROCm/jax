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
# Fast DPX/ROCm validation: GPU preflight + one pytest target.
#
# Defaults: 2 xdist workers/GPU, pmap smoke test on all visible GPUs.
# Override with ROCM_PYTEST_WORKERS_PER_GPU, ROCM_PYTEST_SMOKE_TARGET, and
# ROCM_PYTEST_SMOKE_USE_XDIST (set to 1 for a single-accel test with xdist).
set -exu -o history -o allexport

export ROCM_PYTEST_WORKERS_PER_GPU="${ROCM_PYTEST_WORKERS_PER_GPU:-2}"

source ./ci/utilities/prepare_rocm_tests.sh

smoke_target="${ROCM_PYTEST_SMOKE_TARGET:-tests/pmap_test.py::PythonPmapTest::testBasic}"
use_xdist="${ROCM_PYTEST_SMOKE_USE_XDIST:-0}"

echo "ROCm pytest smoke: target=${smoke_target}"
echo "ROCm pytest smoke: workers_per_gpu=${ROCM_PYTEST_WORKERS_PER_GPU}, num_processes=${num_processes}"

mkdir -p "${LOGS_DIR}" test-artifacts

if [[ "${use_xdist}" == "1" ]]; then
  echo "Running smoke with xdist (-n ${num_processes})..."
  "$JAXCI_PYTHON" -m pytest -n "${num_processes}" --tb=short \
    --json-report --json-report-file="${LOGS_DIR}/pytest_results_smoke.json" \
    --junitxml=test-artifacts/junit-smoke.xml \
    "${smoke_target}"
else
  echo "Running smoke without xdist (multi-GPU / single-process)..."
  unset JAX_ENABLE_ROCM_XDIST
  "$JAXCI_PYTHON" -m pytest --tb=short \
    --json-report --json-report-file="${LOGS_DIR}/pytest_results_smoke.json" \
    --junitxml=test-artifacts/junit-smoke.xml \
    "${smoke_target}"
fi

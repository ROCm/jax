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

# Runs a MaxText ROCm benchmark workload.
#
# This script:
#   - prepares the benchmark environment
#   - executes the workload
#   - collects benchmark run metadata
#   - evaluates benchmark metrics
#   - generates the final ROCm run manifest
set -euo pipefail

WORKLOAD="${1:-gemma3_4b}"

JAX_DIR="${JAX_DIR:-$PWD}"

PYTHON="${JAXCI_PYTHON:-python3}"
PYTHON_VERSION="${JAXCI_HERMETIC_PYTHON_VERSION:-3.12}"

TARGET="maxtext_rocm"
TARGET_DIR="${JAX_DIR}/ci/benchmark_targets/${TARGET}"

MAXTEXT="${TARGET_DIR}/maxtext"
RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"

mkdir -p "${RUN_DIR}"

source "${JAX_DIR}/ci/envs/default.env"
source "${JAX_DIR}/ci/utilities/install_wheels_locally.sh"

if [[ ! -d "${MAXTEXT}/.git" ]]; then
  git clone \
    --depth 1 \
    --branch add-rocm-benchmark-configs \
    https://github.com/ROCm/maxtext.git \
    "${MAXTEXT}"
fi

for file in \
  "${MAXTEXT}/src/maxtext/configs/gpu/models/${WORKLOAD}-rocm.yml" \
  "${MAXTEXT}/src/dependencies/requirements/requirements_rocm_benchmark.txt" \
  "${TARGET_DIR}/baseline.yml"; do
  [[ -f "${file}" ]] || {
    echo "missing required file: ${file}" >&2
    exit 2
  }
done

"${PYTHON}" -m pip install -r \
  "${MAXTEXT}/src/dependencies/requirements/requirements_rocm_benchmark.txt"

export PY_COLORS=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0

export JAX_ENABLE_X64="${JAXCI_ENABLE_X64:-0}"

export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e

pushd "${MAXTEXT}/src" >/dev/null

"${PYTHON}" -m maxtext.trainers.pre_train.train \
  "$(realpath "maxtext/configs/gpu/models/${WORKLOAD}-rocm.yml")" \
> "${RUN_DIR}/run.log" 2>&1

RUN_CODE=$?

popd >/dev/null

set -e

RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

"${PYTHON}" "${JAX_DIR}/ci/collect_bench_manifest_rocm.py" \
  --target "${TARGET}" \
  --workload "${WORKLOAD}" \
  --run-code "${RUN_CODE}" \
  --run-started-at "${RUN_STARTED_AT}" \
  --run-completed-at "${RUN_COMPLETED_AT}" \
  --raw workload_config="${MAXTEXT}/src/maxtext/configs/gpu/models/${WORKLOAD}-rocm.yml" \
  --raw requirements="${MAXTEXT}/src/dependencies/requirements/requirements_rocm_benchmark.txt" \
  --raw baseline_config="${TARGET_DIR}/baseline.yml" \
  --out "${RUN_DIR}/benchmark_meta.json"

CMP_CODE=0

"${PYTHON}" "${TARGET_DIR}/cmp_maxtext_rocm.py" \
  --log "${RUN_DIR}/run.log" \
  --baseline "${TARGET_DIR}/baseline.yml" \
  --workload "${WORKLOAD}" \
  --out "${RUN_DIR}/benchmark_metrics.json" || CMP_CODE=$?

"${PYTHON}" "${JAX_DIR}/ci/collect_run_manifest_rocm.py" \
  --runner "${INPUT_RUNNER}" \
  --python-version "${PYTHON_VERSION}" \
  --python-bin "${PYTHON}" \
  --rocm-version "${INPUT_ROCM_VERSION}" \
  --rocm-tag "${INPUT_ROCM_TAG}" \
  --extra "${RUN_DIR}/benchmark_meta.json" \
  --extra "${RUN_DIR}/benchmark_metrics.json" \
  --out "${RUN_DIR}/result.json"

rm -f \
  "${RUN_DIR}/run.log" \
  "${RUN_DIR}/benchmark_meta.json" \
  "${RUN_DIR}/benchmark_metrics.json"

[[ -s "${RUN_DIR}/result.json" ]] && touch "${RUN_DIR}/_SUCCESS"

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))

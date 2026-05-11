#!/usr/bin/env bash
# Runs a MaxText ROCm benchmark workload and produces benchmark-specific
# result metadata for inclusion in the final ROCm CI run manifest.
#
# The benchmark result payload is written to benchmark.json and later
# merged into the final result.json by collect_rocm_run_metadata.py.
#
# This script:
#   - installs MaxText benchmark dependencies
#   - optionally installs Transformer Engine
#   - runs the benchmark workload
#   - evaluates benchmark results against expected thresholds
#   - writes benchmark-specific metadata to benchmark.json
#
# The final ROCm CI result manifest is produced separately by
# collect_rocm_run_metadata.py.
set -euo pipefail

WORKLOAD="${1:-gemma3-4b}"

JAX_DIR="${JAX_DIR:-$PWD}"

PYTHON_BIN="${JAXCI_PYTHON:-python3}"
PYTHON_VERSION="${JAXCI_HERMETIC_PYTHON_VERSION:-3.12}"
JAX_ENABLE_X64="${JAXCI_ENABLE_X64:-0}"
USE_TE="${USE_TE:-0}"

TARGET_DIR="${JAX_DIR}/ci/benchmark_targets/maxtext_rocm"
MAXTEXT_DIR="${TARGET_DIR}/maxtext"
MAXTEXT_SRC_DIR="${MAXTEXT_DIR}/src"

RUN_DIR="${TARGET_DIR}/run_artifacts/${WORKLOAD}"
RUN_LOG="${RUN_DIR}/model_run.log"
BENCH_JSON="${RUN_DIR}/benchmark.json"
RESULT_JSON="${RUN_DIR}/result.json"

CFG_FILE="${MAXTEXT_DIR}/src/maxtext/configs/gpu/models/${WORKLOAD}-rocm.yml"
REQ_FILE="${MAXTEXT_DIR}/src/dependencies/requirements/requirements_rocm_benchmark.txt"
EXP_FILE="${TARGET_DIR}/exp_maxtext_rocm.yml"

mkdir -p "${RUN_DIR}"

source "${JAX_DIR}/ci/envs/default.env"
source "${JAX_DIR}/ci/utilities/install_wheels_locally.sh"

if [[ ! -d "${MAXTEXT_DIR}/.git" ]]; then
  git clone \
    --depth 1 \
    --branch add-rocm-benchmark-configs \
    https://github.com/ROCm/maxtext.git \
    "${MAXTEXT_DIR}"
fi

[[ -f "${CFG_FILE}" ]] || {
  echo "missing config file: ${CFG_FILE}" >&2
  exit 2
}

[[ -f "${REQ_FILE}" ]] || {
  echo "missing requirements file: ${REQ_FILE}" >&2
  exit 2
}

[[ -f "${EXP_FILE}" ]] || {
 echo "missing expected file: ${EXP_FILE}" >&2
 exit 2
}

echo "Installing MaxText ROCm benchmark requirements"
"${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"

if [[ "${USE_TE}" == "1" ]]; then
  echo "Resolving latest Transformer Engine wheel"

  PY_TAG="cp$(echo "${PYTHON_VERSION}" | tr -d '.')"

  TE_WHEEL_URL="$(
    curl -fsSL https://api.github.com/repos/ROCm/maxtext/releases \
      | grep "browser_download_url" \
      | grep "te-rocm-wheels-" \
      | grep "${PY_TAG}" \
      | head -n1 \
      | cut -d '"' -f4
  )"

  [[ -n "${TE_WHEEL_URL}" ]] || {
    echo "Failed to resolve Transformer Engine wheel" >&2
    exit 1
  }

  echo "Installing Transformer Engine from ${TE_WHEEL_URL}"
  "${PYTHON_BIN}" -m pip install --no-deps "${TE_WHEEL_URL}"
fi

export PY_COLORS=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="${JAXCI_ENABLE_X64}"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo "Running MaxText workload: ${WORKLOAD}.."

set +e
pushd "${MAXTEXT_SRC_DIR}" >/dev/null

"${PYTHON_BIN}" -m maxtext.trainers.pre_train.train \
  "$(realpath "${CFG_FILE}")" \
> "${RUN_LOG}" 2>&1

RUN_CODE=$?

popd >/dev/null
set -e

echo "..Completed MaxText workload: ${WORKLOAD}"

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

CMP_CODE=0

"${PYTHON_BIN}" "${TARGET_DIR}/cmp_maxtext_rocm.py" \
  --log "${RUN_LOG}" \
  --expected "${EXP_FILE}" \
  --config "${CFG_FILE}" \
  --requirements "${REQ_FILE}" \
  --target maxtext_rocm \
  --workload "${WORKLOAD}" \
  --run-code "${RUN_CODE}" \
  --model-run-started-at "${MODEL_RUN_STARTED_AT}" \
  --model-run-completed-at "${MODEL_RUN_COMPLETED_AT}" \
  --out "${BENCH_JSON}" || CMP_CODE=$?

"${PYTHON_BIN}" "${JAX_DIR}/ci/collect_run_manifest_rocm.py" \
  --runner "${INPUT_RUNNER}" \
  --python-version "${PYTHON_VERSION}" \
  --python-bin "${PYTHON_BIN}" \
  --rocm-version "${INPUT_ROCM_VERSION}" \
  --rocm-tag "${INPUT_ROCM_TAG}" \
  --extra "${BENCH_JSON}" \
  --out "${RESULT_JSON}"

rm -f "${RUN_LOG}" "${BENCH_JSON}"

[[ -s "${RESULT_JSON}" ]] && touch "${RUN_DIR}/_SUCCESS"

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))

#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="gemma3-4b"
[[ $# -eq 0 || ( $# -eq 2 && "$1" == "--workload" ) ]] || {
  echo "usage: $0 [--workload NAME]" >&2
  exit 2
}
[[ $# -eq 2 ]] && WORKLOAD="$2"

ROOT="${ROOT:-$PWD}"
TARGET_DIR="${ROOT}/benchmark_targets/rocm_maxtext"
RUN_DIR="${ROOT}/run_artifacts"
REPO_DIR="${ROOT}/maxtext"
WORK_DIR="${REPO_DIR}/src"

CFG_FILE="${REPO_DIR}/configs/models/${WORKLOAD}.yml"
ENV_FILE="${REPO_DIR}/configs/models/${WORKLOAD}.env.sh"
REQ_FILE="${WORK_DIR}/dependencies/requirements/requirements_rocm_jax_0.8.2.txt"
EXPECTED_FILE="${TARGET_DIR}/expected.json"

RUN_LOG="${RUN_DIR}/maxtext.log"
BENCHMARK_JSON="${RUN_DIR}/benchmark.json"
RESULT_JSON="${RUN_DIR}/result.json"

PYTHON_BIN="${JAXCI_PYTHON:-python3}"

mkdir -p "${RUN_DIR}"

source ci/envs/default.env
source ./ci/utilities/install_wheels_locally.sh

[[ -d "${REPO_DIR}/.git" ]] || \
  git clone --depth 1 --branch main https://github.com/ROCm/maxtext.git "${REPO_DIR}"

[[ -f "${CFG_FILE}" ]] || { echo "missing config file: ${CFG_FILE}" >&2; exit 2; }
[[ -f "${EXPECTED_FILE}" ]] || { echo "missing expected file: ${EXPECTED_FILE}" >&2; exit 2; }

[[ -f "${REQ_FILE}" ]] && {
  echo "[setup] installing MaxText requirements"
  "${PYTHON_BIN}" -m pip install -r "${REQ_FILE}"
}

[[ -f "${ENV_FILE}" ]] && source "${ENV_FILE}"

if [[ "${USE_TE:-0}" == "1" ]]; then
  echo "[setup] resolving latest Transformer Engine wheel"
  PY_TAG="cp$(echo "${JAXCI_HERMETIC_PYTHON_VERSION:-3.12}" | tr -d '.')"
  TE_WHEEL_URL="$(
    curl -fsSL https://api.github.com/repos/ROCm/maxtext/releases \
      | grep "browser_download_url:" \
      | grep "te-rocm-wheels-" \
      | grep "${PY_TAG}" \
      | grep linux_x86_64.whl \
      | head -n1 \
      | cut -d '"' -f4
  )"

  [[ -n "${TE_WHEEL_URL}" ]] || {
    echo "failed to resolve Transformer Engine wheel for ${PY_TAG}" >&2
    exit 1
  }
  
  echo "[setup] installing Transformer Engine from ${TE_WHEEL_URL}"
  "${PYTHON_BIN}" -m pip install --no-deps "${TE_WHEEL_URL}"
fi

export PY_COLORS=1
export NCCL_DEBUG=WARN
export TF_CPP_MIN_LOG_LEVEL=0
export JAX_ENABLE_X64="${JAXCI_ENABLE_X64}"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_PYTHON_CLIENT_PREALLOCATE=false

MODEL_RUN_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

set +e
pushd "${WORK_DIR}" >/dev/null
"${PYTHON_BIN}" -m maxtext.trainers.pre_train.train \
  "$(realpath "${CFG_FILE}")" \
> "${RUN_LOG}" 2>&1
RUN_CODE=$?
popd >/dev/null
set -e

MODEL_RUN_COMPLETED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

CMP_CODE=1
"${PYTHON_BIN}" "${TARGET_DIR}/benchmark_result.py" \
  --log "${RUN_LOG}" \
  --expected "${EXPECTED_FILE}" \
  --config "${CFG_FILE}" \
  --env-file "${ENV_FILE}" \
  --requirements "${REQ_FILE}" \
  --target rocm_maxtext \
  --workload "${WORKLOAD}" \
  --run-code "${RUN_CODE}" \
  --model-run-started-at "${MODEL_RUN_STARTED_AT}" \
  --model-run-completed-at "${MODEL_RUN_COMPLETED_AT}" \
  --out "${BENCHMARK_JSON}" || CMP_CODE=$?

"${PYTHON_BIN}" ci/make_ci_manifest.py \
  --extra "${BENCHMARK_JSON}" \
  --out "${RESULT_JSON}"

rm -f "${RUN_LOG}" "${BENCHMARK_JSON}"

[[ -s "${RESULT_JSON}" ]] && touch "${RUN_DIR}/_SUCCESS"

exit $(( RUN_CODE != 0 || CMP_CODE != 0 ))
#!/usr/bin/env bash
set -euo pipefail

: "${S3_BUCKET_NAME:?}"
ARTIFACT_DIR="${1:-run_artifacts}"
RESULT_FILE="${ARTIFACT_DIR}/result.json"

[[ -f "${RESULT_FILE}" ]] || {
  echo "missing result.json" >&2
  exit 2
}

RUN_KEY="${DATE}_${GITHUB_RUN_ID}_${GITHUB_RUN_ATTEMPT}"
COMBO="py$(norm "${INPUT_PYTHON}")-rocm$(norm "${INPUT_ROCM_VERSION}")-${GPU_PART}"
PREFIX="${GITHUB_REPOSITORY}/${GITHUB_REF_NAME}/${IS_NIGHTLY}/${RUN_KEY}/${COMBO}"

DEST="s3://${S3_BUCKET_NAME}/${TEST_LOGS_ROOT}/${PREFIX}"

echo "[upload] ${DEST}"

aws s3 cp --only-show-errors "${RESULT_FILE}" "${DEST}/result.json"

if [[ -f "${ARTIFACT_DIR}/logs.tar.gz" ]]; then
  aws s3 cp --only-show-errors "${ARTIFACT_DIR}/logs.tar.gz" "${DEST}/logs.tar.gz"
fi

printf '' | aws s3 cp --only-show-errors - "${DEST}/_SUCCESS"

echo "[done]"
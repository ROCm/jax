#!/usr/bin/env bash
set -euo pipefail

# Uploads a file or directory to S3.
#
# Usage:
#   upload_rocm_artifacts.sh <source_path> <s3_dest_prefix> [success]
#
# If the optional third argument is "success", an empty _SUCCESS marker is
# written after the upload completes.

: "${S3_BUCKET_NAME:?}"

SRC="${1:?missing source path}"
DEST_PREFIX="${2:?missing S3 destination prefix}"
WRITE_SUCCESS="${3:-}"

[[ -e "${SRC}" ]] || {
 echo "missing source path: ${SRC}" >&2
 exit 2
}

DEST="s3://${S3_BUCKET_NAME}/${DEST_PREFIX}"

echo "[upload] ${SRC} -> ${DEST}"

if [[ -d "${SRC}" ]]; then
  aws s3 cp --only-show-errors "${SRC}" "${DEST}" --recursive
else
  aws s3 cp --only-show-errors "${SRC}" "${DEST}/$(basename "${SRC}")"
fi

if [[ "${WRITE_SUCCESS}" == "success" ]]; then
  printf '' | aws s3 cp --only-show-errors - "${DEST}/_SUCCESS"
fi

echo "[done]"

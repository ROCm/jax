#!/bin/bash

set -e
set -x

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc=$SCRIPT_DIR/rocm.bazelrc test \
    --config=rocm \
    --repo_env="ROCM_DISTRO_VERSION=rocm_7.10.0_gfx90X" \
    --build_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --test_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942 \
    --//jax:build_jaxlib=true \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    --test_filter='CoreTest|JaxprTypeChecks|DynamicShapesTest|testMatmul' \
    $@ \
    -- \
    //tests:core_test_gpu \
    //tests:linalg_test_gpu \
    //tests:ffi_test_gpu \
    //tests:linalg_test_gpu \
    //tests:ffi_test_gpu \

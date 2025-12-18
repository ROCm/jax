#!/bin/bash

set -e
set -x

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc=$SCRIPT_DIR/rocm.bazelrc run \
    --config=rocm \
    --repo_env="ROCM_DISTRO_VERSION=rocm_7.10.0_gfx90X" \
    --build_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --test_tag_filters=cpu,gpu,-tpu,-config-cuda-only \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942 \
    --//jax:build_jaxlib=true \
    --action_env=TF_ROCM_AMDGPU_TARGETS="gfx906,gfx908,gfx90a,gfx942,gfx950,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201" \
    --test_verbose_timeout_warnings \
    --test_output=errors \
    $@

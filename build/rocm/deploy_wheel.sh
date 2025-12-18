#!/bin/bash

set -e
set -x

SCRIPT_DIR=$(dirname $0)
bazel --bazelrc=$SCRIPT_DIR/rocm.bazelrc run \
    --config=rocm \
    --repo_env="ROCM_DISTRO_VERSION=rocm_7.10.0_gfx90X" \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908,gfx90a,gfx942 \
    --//jax:build_jaxlib=true \
    --sandbox_debug \
    --verbose_failures \
    --repo_env=HERMETIC_PYTHON_VERSION=3.12 \
    //build/rocm:deploy_jax_wheel $1

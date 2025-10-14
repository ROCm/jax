#!/bin/bash

set -e

SCRIPT_DIR=$(dirname $0)
pushd $SCRIPT_DIR

python build/build.py build --wheels=jax-rocm-plugin --configure_only --python_version=3.12
./bazel-7.4.1-linux-x86_64 test \
    --config=rocm \
    --action_env=TF_ROCM_AMDGPU_TARGETS=gfx908 \
    --//jax:build_jaxlib=true \
    --test_output=streamed \
    "//tests:core_test_gpu"

popd

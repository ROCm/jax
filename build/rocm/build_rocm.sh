#!/usr/bin/env bash

set -eux

ROCM_TF_FORK_REPO="https://github.com/ROCmSoftwarePlatform/tensorflow-upstream"
ROCM_TF_FORK_BRANCH="jax_rocm-4.5"
ls /opt
git clone -b ${ROCM_TF_FORK_BRANCH} ${ROCM_TF_FORK_REPO}

python3 ./build/build.py --enable_rocm --rocm_path=${ROCM_PATH} --bazel_options=--override_repository=org_tensorflow=/workspace/tensorflow-upstream
pip3 install --use-feature=2020-resolver --force-reinstall dist/*.whl  # installs jaxlib (includes XLA)
pip3 install --use-feature=2020-resolver --force-reinstall .  # installs jax

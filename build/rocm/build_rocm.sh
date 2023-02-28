#!/usr/bin/env bash
# Copyright 2022 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Environment Var Notes
# XLA_CLONE_DIR	-
#	Specifies filepath to where XLA repo is cloned. 
#	NOTE:, if this is set then XLA repo is not cloned. Must clone repo before running this script. 
#	Also, if this is set then setting XLA_REPO and XLA_BRANCH have no effect.
# XLA_REPO 
#	XLA repo to clone from. Default is https://github.com/ROCmSoftwarePlatform/tensorflow-upstream
# XLA_BRANCH
#	XLA branch in the XLA repo. Default is develop-upstream-jax
#

set -eux
pyenv local $PYTHON_VERSION
python -V

#If XLA_REPO is not set, then use default
if [ ! -v XLA_REPO ]; then
	XLA_REPO="https://github.com/ROCmSoftwarePlatform/tensorflow-upstream"
	XLA_BRANCH="develop-upstream-jax"
elif [ -z "$XLA_REPO" ]; then
	XLA_REPO="https://github.com/ROCmSoftwarePlatform/tensorflow-upstream"
	XLA_BRANCH="develop-upstream-jax"
fi

#If XLA_CLONE_PATH is not set, then use default path. 
#Note, setting XLA_CLONE_PATH makes setting XLA_REPO and XLA_BRANCH a no-op
#Set this when XLA repository has been already clone. This is useful in CI
#environments and when doing local development
if [ ! -v XLA_CLONE_DIR ]; then
	XLA_CLONE_DIR=/tmp/tensorflow-upstream
	rm -rf /tmp/tensorflow-upstream || true
	git clone -b ${XLA_BRANCH} ${XLA_REPO} /tmp/tensorflow-upstream
elif [ -z "$XLA_CLONE_DIR" ]; then
	XLA_CLONE_DIR=/tmp/tensorflow-upstream
	rm -rf /tmp/tensorflow-upstream || true
	git clone -b ${XLA_BRANCH} ${XLA_REPO} /tmp/tensorflow-upstream
fi

#python3 ./build/build.py --enable_rocm --rocm_path=${ROCM_PATH} --bazel_options=--override_repository=org_tensorflow=${XLA_CLONE_DIR}
#pip3 install --force-reinstall dist/*.whl  # installs jaxlib (includes XLA)
#pip3 install --force-reinstall .  # installs jax

if [ -v JAX_RENAME_WHL ]; then
	if [ -n "$JAX_RENAME_WHL" ]; then
		rocm_version=$(cat /opt/rocm/.info/version)
		rocmv=${rocm_version//./}
		rocmv=${rocmv:0:3}
		owhl=$(basename dist/*.whl)
		nwhl=${owhl:0:12}.$rocmv${owhl:12}
		echo $nwhl
		mv dist/$owhl dist/$nwhl
	fi
fi

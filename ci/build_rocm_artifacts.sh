#!/bin/bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Build ROCm JAX artifacts.
# Usage: ./ci/build_rocm_artifacts.sh "<artifact>"
# Supported artifact values are: jax-rocm-plugin, jax-rocm-pjrt
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# Initialize TheRock SDK if installed via pip wheels.
if command -v rocm-sdk &>/dev/null; then
  rocm-sdk init
fi

artifact="$1"

# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" && -z "$JAXCI_CLONE_MAIN_XLA" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment.
source "ci/utilities/setup_build_environment.sh"

OVERRIDE_XLA_REPO=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  OVERRIDE_XLA_REPO="--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

allowed_artifacts=("jax-rocm-plugin" "jax-rocm-pjrt")

if [[ "$artifact" == "jax-rocm-plugin" ]]; then
  deploy_target="//jaxlib/tools:deploy_rocm_plugin_wheel"
elif [[ "$artifact" == "jax-rocm-pjrt" ]]; then
  deploy_target="//jaxlib/tools:deploy_rocm_pjrt_wheel"
else
  echo "Error: Invalid artifact: $artifact. Allowed values are: ${allowed_artifacts[*]}"
  exit 1
fi

wheel_version_suffix_flag=""
if [[ -n "${JAXCI_WHEEL_VERSION_SUFFIX:-}" ]]; then
  wheel_version_suffix_flag="--action_env=WHEEL_VERSION_SUFFIX=${JAXCI_WHEEL_VERSION_SUFFIX}"
fi

echo "Building $artifact..."

bazel --bazelrc=build/rocm/rocm.bazelrc run \
      --config=rocm_release_wheel \
      --config=rocm_rbe \
      --repo_env=HERMETIC_PYTHON_VERSION="${JAXCI_HERMETIC_PYTHON_VERSION}" \
      $wheel_version_suffix_flag \
      $OVERRIDE_XLA_REPO \
      $deploy_target -- "$JAXCI_OUTPUT_DIR/"

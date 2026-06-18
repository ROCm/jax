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
# ----------------------------------------------------------------------------
# JAXCI_LOCAL_BUILD: CI vs. local (no-RBE) mode
# ----------------------------------------------------------------------------
# By DEFAULT (JAXCI_LOCAL_BUILD unset or set to anything other than "1") this
# script behaves EXACTLY as it always has: it builds the single artifact passed
# as $1 for $JAXCI_HERMETIC_PYTHON_VERSION using the remote build execution
# (RBE) configs `--config=rocm_rbe --remote_download_toplevel`. This is what the
# .github/workflows/build_rocm_artifacts.yml workflow runs (once per matrix
# cell over {plugin,pjrt} x {python versions}).
#
# Setting JAXCI_LOCAL_BUILD=1 switches to a local mirror of that build that runs
# entirely on the local machine. The ONLY build difference is that the RBE
# configs are omitted. Every other `build/build.py` flag is identical to CI.
# Local mode ignores $1 and builds the full set of wheels CI produces
# across its matrix: the pjrt wheel once (for the first Python version) and
# the plugin wheel for every Python version.
#
# XLA is controlled by the usual CI vars in both modes (JAXCI_XLA_GIT_DIR,
# JAXCI_CLONE_MAIN_XLA, JAXCI_XLA_COMMIT). With none of them set, the
# WORKSPACE-pinned XLA is used (local never auto-clones since, unlike the
# workflow, nothing sets JAXCI_CLONE_MAIN_XLA=1).
#
# Local mode must be run from the jax repo root (like CI). Wheels are written to
# jax/dist.
#
# Local-mode-only env var overrides (CI-faithful defaults):
#   PYTHON_VERSIONS         comma-separated CPython versions. The plugin wheel is
#                           built for each; the pjrt wheel is built once for the
#                           first version. [default: 3.11,3.12,3.13,3.14]
#   RUN_AUDITWHEEL          1 to run ci/utilities/run_auditwheel.sh [default: 1]
#
# The artifact type (default/nightly/release) is taken from JAXCI_ARTIFACT_TYPE
# in both modes, exactly as before.
#
# -e: abort script if one command fails
# -u: error if undefined variable used
# -x: log all commands
# -o history: record shell history
# -o allexport: export all functions and variables to be available to subscripts
set -exu -o history -o allexport

# When "1", run the local (no-RBE) build path. Any other value (including
# unset) keeps the original CI behavior unchanged.
JAXCI_LOCAL_BUILD="${JAXCI_LOCAL_BUILD:-0}"

artifact="${1:-}"

# ----------------------------------------------------------------------------
# Local-mode preamble. Skipped entirely in CI mode so CI behavior is unchanged.
# ----------------------------------------------------------------------------
if [[ "$JAXCI_LOCAL_BUILD" == 1 ]]; then
  # Local builds run from the repo root, just like CI (the relative
  # `source ci/...` calls below depend on it).
  if [[ ! -f build/build.py ]]; then
    echo "ERROR: build/build.py not found. Run this script from the jax repo root." >&2
    exit 1
  fi

  PYTHON_VERSIONS="${PYTHON_VERSIONS:-3.11,3.12,3.13,3.14}"
  IFS=',' read -ra PY_ARR <<< "$PYTHON_VERSIONS"
  RUN_AUDITWHEEL="${RUN_AUDITWHEEL:-1}"
fi

# Source default JAXCI environment variables.
source ci/envs/default.env

# Clone XLA at HEAD if path to local XLA is not provided
if [[ -z "$JAXCI_XLA_GIT_DIR" && -z "$JAXCI_CLONE_MAIN_XLA" ]]; then
    export JAXCI_CLONE_MAIN_XLA=1
fi

# Set up the build environment (clones/overrides XLA, checks bazel, mkdir output).
source "ci/utilities/setup_build_environment.sh"

# CI builds a single artifact passed as $1; validate it. Local mode ignores $1
# and always builds the full set, so no artifact validation is needed.
if [[ "$JAXCI_LOCAL_BUILD" != 1 ]]; then
  allowed_artifacts=("jax-rocm-plugin" "jax-rocm-pjrt")
  if [[ ! " ${allowed_artifacts[*]} " =~ " ${artifact} " ]]; then
    echo "Error: Invalid artifact: $artifact. Allowed values are: ${allowed_artifacts[*]}"
    exit 1
  fi
fi

# Determine the artifact tag flags based on the artifact type, mirroring
# ci/build_artifacts.sh.
if [[ "$JAXCI_ARTIFACT_TYPE" == "release" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=release --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
elif [[ "$JAXCI_ARTIFACT_TYPE" == "nightly" ]]; then
  current_date=$(date +%Y%m%d)
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=${current_date} --bazel_options=--repo_env=ML_WHEEL_TYPE=nightly --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
elif [[ "$JAXCI_ARTIFACT_TYPE" == "default" ]]; then
  artifact_tag_flags="--bazel_options=--repo_env=ML_WHEEL_TYPE=custom --bazel_options=--repo_env=ML_WHEEL_BUILD_DATE=$(git show -s --format=%as HEAD) --bazel_options=--repo_env=ML_WHEEL_GIT_HASH=$(git rev-parse HEAD) --bazel_options=--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)"
else
  echo "Error: Invalid artifact type: $JAXCI_ARTIFACT_TYPE. Allowed values are: release, nightly, default"
  exit 1
fi

override_xla_repo=""
if [[ "$JAXCI_CLONE_MAIN_XLA" == 1 ]]; then
  override_xla_repo="--bazel_options=--override_repository=xla=${JAXCI_XLA_GIT_DIR}"
fi

wheel_version_suffix_flag=""
if [[ -n "${JAXCI_WHEEL_VERSION_SUFFIX:-}" ]]; then
  wheel_version_suffix_flag="--bazel_options=--repo_env=ML_WHEEL_VERSION_SUFFIX=${JAXCI_WHEEL_VERSION_SUFFIX}"
fi

bazel_startup_options=""
if [[ -n "${JAXCI_BAZEL_OUTPUT_BASE}" ]]; then
  bazel_startup_options="--bazel_startup_options=--output_base=${JAXCI_BAZEL_OUTPUT_BASE}"
fi

# Point the build ROCm at the TheRock SDK already installed in this image, so
# the wheel's ROCm SONAMEs (e.g. librocsolver) match the runtime image. Without
# this, rules_ml_toolchain silently downloads its default distro (a different,
# older ROCm) and the wheel links e.g. librocsolver.so.1 while the runtime only
# has librocsolver.so.0. rules_ml_toolchain honors ROCM_PATH ahead of its
# download path; rocm-sdk path --root is the pip-installed SDK root (no
# /opt/rocm needed). Gated on rocm-sdk so apt-ROCm images keep the default.
rocm_path_flags=""
if command -v rocm-sdk >/dev/null 2>&1; then
  rocm_path_flags="--bazel_options=--repo_env=ROCM_PATH=$(rocm-sdk path --root)"
fi

# In CI mode add the RBE configs. In local mode they are omitted so the build
# runs on the local machine.
rbe_flags=""
if [[ "$JAXCI_LOCAL_BUILD" != 1 ]]; then
  rbe_flags="--bazel_options=--config=rocm_rbe --bazel_options=--remote_download_toplevel"
fi

# Single build invocation. Identical between CI and local modes except that the
# RBE configs ($rbe_flags) are empty in local mode.
build_artifact() {
  local artifact="$1" py="$2"
  echo "Building $artifact for Python $py..."
  python build/build.py build --wheels="$artifact" \
    --bazel_startup_options="--bazelrc=build/rocm/rocm.bazelrc" \
    $bazel_startup_options \
    --bazel_options=--config=rocm_release_wheel \
    $rbe_flags \
    --python_version="$py" \
    --verbose --detailed_timestamped_log \
    --output_path="$JAXCI_OUTPUT_DIR" \
    $artifact_tag_flags \
    $override_xla_repo \
    $wheel_version_suffix_flag \
    $rocm_path_flags
}

if [[ "$JAXCI_LOCAL_BUILD" == 1 ]]; then
  # Local mode reproduces CI's full matrix output: the pjrt wheel once (for the
  # first Python version) and the plugin wheel for every Python version.
  build_artifact jax-rocm-pjrt "${PY_ARR[0]}"
  for py in "${PY_ARR[@]}"; do
    build_artifact jax-rocm-plugin "$py"
  done
else
  # CI: build the single requested artifact for the hermetic Python version.
  build_artifact "$artifact" "$JAXCI_HERMETIC_PYTHON_VERSION"
fi

# Verify manylinux compliance.
./ci/utilities/run_auditwheel.sh

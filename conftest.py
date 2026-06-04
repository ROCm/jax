# Copyright 2021 The JAX Authors.
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
"""pytest configuration"""

import os
import sys
import pytest


@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
  import jax
  import numpy

  doctest_namespace["jax"] = jax
  doctest_namespace["lax"] = jax.lax
  doctest_namespace["jnp"] = jax.numpy
  doctest_namespace["np"] = numpy


# A pytest hook that runs immediately before test collection (i.e. when pytest
# loads all the test cases to run). When running parallel tests via xdist on
# GPU or Cloud TPU, we use this hook to set the env vars needed to run multiple
# test processes across different chips.
#
# It's important that the hook runs before test collection, since jax tests end
# up initializing the TPU runtime on import (e.g. to query supported test
# types). It's also important that the hook gets called by each xdist worker
# process. Luckily each worker does its own test collection.
#
# The pytest_collection hook can be used to overwrite the collection logic, but
# we only use it to set the env vars and fall back to the default collection
# logic by always returning None. See
# https://docs.pytest.org/en/latest/how-to/writing_hook_functions.html#firstresult-stop-at-first-non-none-result
# for details.
#
# For TPU, the env var JAX_ENABLE_TPU_XDIST must be set for this hook to have an
# effect. We do this to minimize any effect on non-TPU tests, and as a pointer
# in test code to this "magic" hook. TPU tests should not specify more xdist
# workers than the number of TPU chips.
#
# For GPU, the env var JAX_ENABLE_CUDA_XDIST must be set equal to the number of
# CUDA devices. Test processes will be assigned in round robin fashion across
# the devices.
def pytest_collection() -> None:
  if os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault("TPU_VISIBLE_CHIPS", str(xdist_worker_number))
    os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")

  elif num_cuda_devices := os.environ.get("JAX_ENABLE_CUDA_XDIST", None):
    num_cuda_devices = int(num_cuda_devices)
    # When running as an xdist worker, will be something like "gw0"
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])
    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES", str(xdist_worker_number % num_cuda_devices)
    )

  elif num_rocm_devices := os.environ.get("JAX_ENABLE_ROCM_XDIST", None):
    num_rocm_devices = int(num_rocm_devices)
    xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "")
    if not xdist_worker_name.startswith("gw"):
      return
    xdist_worker_number = int(xdist_worker_name[len("gw") :])

    # The CI runner isolates this job to a subset of the host's physical GPUs
    # by injecting ROCR_VISIBLE_DEVICES via the container env-file (e.g.
    # "0,3,4,5"). Those are absolute ROCt/physical indices and the set may be
    # non-contiguous. ROCR_VISIBLE_DEVICES does not compose across processes:
    # re-setting it is re-evaluated against the full physical enumeration, not
    # against the inherited slice. So selecting an absolute index of our own
    # (xdist_worker_number % num_rocm_devices) can land on a GPU owned by a
    # co-tenant job on the same host. Instead, pick a device *from the
    # inherited slice* so we never escape the isolation boundary.
    allocated = os.environ.get("ROCR_VISIBLE_DEVICES")
    allocated_tokens = allocated.split(",") if allocated else []

    # Opt-in escape hatch to reproduce the historical absolute-index behaviour,
    # so a contrasting CI run can demonstrate an isolation escape on sliced
    # runners. Defaults to the safe slice-relative selection.
    selection_mode = os.environ.get("JAX_ROCM_XDIST_SELECT", "slice")

    if selection_mode == "absolute" or not allocated_tokens:
      selected = str(xdist_worker_number % num_rocm_devices)
    else:
      selected = allocated_tokens[xdist_worker_number % len(allocated_tokens)]

    within_isolation = (not allocated_tokens) or (selected in allocated_tokens)

    os.environ["ROCR_VISIBLE_DEVICES"] = selected
    # ROCr now surfaces a single physical device, which becomes HIP index 0.
    # The container env-file may preset HIP_VISIBLE_DEVICES to all GPUs; pin to
    # "0" so HIP doesn't try to enable agents that ROCr just hid.
    os.environ["HIP_VISIBLE_DEVICES"] = "0"

    status = "WITHIN-ISOLATION" if within_isolation else "ISOLATION-ESCAPE"
    print(
        f"[rocm-xdist] {xdist_worker_name}: mode={selection_mode} "
        f"runner_allocated_ROCR={allocated_tokens or 'unset'} "
        f"selected_ROCR={selected} HIP_VISIBLE_DEVICES=0 -> {status}",
        file=sys.stderr,
        flush=True,
    )
    if not within_isolation:
      print(
          f"[rocm-xdist] WARNING {xdist_worker_name}: selected physical device "
          f"{selected!r} is NOT in the runner-allocated set {allocated_tokens}; "
          f"this worker is running on a GPU owned by another job.",
          file=sys.stderr,
          flush=True,
      )

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
import pytest

# Mosaic GPU checking based on test *file path* only (avoid test-name substrings).
_MOSAIC_GPU_PATH_NEEDLES = (
    f"{os.sep}tests{os.sep}mosaic{os.sep}",
    f"{os.sep}tests{os.sep}pallas{os.sep}mgpu_",
    f"{os.sep}tests{os.sep}pallas{os.sep}mosaic_gpu",
    f"{os.sep}tests{os.sep}pallas{os.sep}mosaic",
)

# Simple Mosaic GPU *usage* substring checks (avoid import-only signals).
_MOSAIC_GPU_SOURCE_NEEDLES = (
    "inline_mgpu",
    "plgpu_mgpu.",
    "mosaic_gpu_interpret",
    "mosaic_gpu_backend",
    "jax.experimental.mosaic.gpu",  # runtime usage in body (not module import scan)
    "jax.experimental.pallas.mosaic_gpu",
)


def _pallas_defaults_to_mosaic_gpu() -> bool:
  """Returns True if Pallas GPU lowering defaults to Mosaic GPU."""
  try:
    from jax._src.pallas import pallas_call as pallas_call_lib  # pytype: disable=import-error
    return bool(pallas_call_lib._PALLAS_USE_MOSAIC_GPU.value)  # pylint: disable=protected-access
  except Exception:
    return False


def _running_on_rocm() -> bool:
  """Best-effort ROCm detection.

  First tries to check rocm in jaxlib version, falls back to checking backend 
  platform_version so that it works for ROCm PJRT plugin installs where jaxlib's 
  version tag may not contain rocm.
  """
  try:
    import jaxlib.version as jaxlib_version  # pytype: disable=import-error
    version_str = getattr(jaxlib_version, "__version__", "")
  except Exception:
    version_str = ""
  if "rocm" in version_str.lower():
    return True
  try:
    import jax  # pytype: disable=import-error
    from jax._src import xla_bridge  # pytype: disable=import-error
    backend = xla_bridge.get_backend()
    pv = getattr(backend, "platform_version", "") or ""
    return "rocm" in str(pv).lower()
  except Exception:
    return False


def _source_mentions_mosaic_gpu(src: str) -> bool:
  """Returns True if the test file content has Mosaic GPU usage."""
  lowered = src.lower()
  return any(n in lowered for n in _MOSAIC_GPU_SOURCE_NEEDLES)


def _looks_like_mosaic_gpu_path(path_str: str) -> bool:
  """Returns True if the path is a Mosaic-GPU-only test file."""
  lowered = path_str.lower()
  return any(n.lower() in lowered for n in _MOSAIC_GPU_PATH_NEEDLES)


def _class_mosaic_override(cls: type | None, cache: dict[object, object]) -> bool | None:
  """Detects explicit class-level Mosaic enable/disable.

  Returns:
    - True if the class forces Mosaic GPU (`_PALLAS_USE_MOSAIC_GPU(True)`).
    - False if it forces Triton (`_PALLAS_USE_MOSAIC_GPU(False)`).
    - None if no explicit override is found.
  """
  if cls is None:
    return None
  cache_key = ("__mosaic_override__", cls)
  if cache_key in cache:
    return cache[cache_key]  # type: ignore[return-value]
  import inspect
  try:
    src = inspect.getsource(cls).lower()
  except Exception:
    cache[cache_key] = None
    return None
  if "_pallas_use_mosaic_gpu(true" in src:
    cache[cache_key] = True
    return True
  if "_pallas_use_mosaic_gpu(false" in src:
    cache[cache_key] = False
    return False
  cache[cache_key] = None
  return None


def _is_mosaic_gpu_item(
    item: pytest.Item,
    cache: dict[object, bool],
    *,
    running_on_rocm: bool,
    pallas_defaults_to_mosaic: bool,
) -> bool:
  """Returns True if this test item uses (or would use) Mosaic GPU."""
  path_obj = getattr(item, "path", None) or getattr(item, "fspath", None)
  path_str = str(path_obj) if path_obj is not None else ""
  if _looks_like_mosaic_gpu_path(path_str):
    return True

  import inspect

  obj = getattr(item, "obj", None)
  if obj is None:
    return False
  if obj in cache:
    return cache[obj]
  try:
    src = inspect.getsource(obj)
  except Exception:
    cache[obj] = False
    return False

  lowered = src.lower()
  # Direct Mosaic usage in the test function/method.
  if _source_mentions_mosaic_gpu(lowered):
    cache[obj] = True
    return True

  # Respect explicit class-level override: if a test class forces Mosaic off,
  # we should not skip it just because Pallas defaults to Mosaic elsewhere.
  cls_override = _class_mosaic_override(getattr(item, "cls", None), cache)  # type: ignore[arg-type]
  if cls_override is False:
    cache[obj] = False
    return False
  if cls_override is True:
    cache[obj] = True
    return True

  # Implicit Mosaic usage: on ROCm, `pallas_call` defaults to Mosaic GPU when
  # `compiler_params` is not specified and Mosaic is the default backend.
  if running_on_rocm and pallas_defaults_to_mosaic:
    uses_pallas_call = (
        ".pallas_call" in lowered
        or "pl.pallas_call" in lowered
        or "pallas_call(" in lowered
    )
    explicitly_selects_compiler = "compiler_params=" in lowered
    if uses_pallas_call and not explicitly_selects_compiler:
      cache[obj] = True
      return True

  cache[obj] = False
  return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
  """Mark Mosaic GPU tests and skip them on ROCm."""
  running_on_rocm = _running_on_rocm()
  pallas_defaults_to_mosaic = _pallas_defaults_to_mosaic_gpu() if running_on_rocm else False
  cache: dict[object, bool] = {}
  for item in items:
    is_mosaic_gpu = _is_mosaic_gpu_item(
        item,
        cache,
        running_on_rocm=running_on_rocm,
        pallas_defaults_to_mosaic=pallas_defaults_to_mosaic,
    )
    if not is_mosaic_gpu:
      continue
    item.add_marker(pytest.mark.mosaic_gpu)
    if running_on_rocm:
      item.add_marker(pytest.mark.skip(
          reason="Mosaic GPU tests are not supported on ROCm"
      ))


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
    assigned = str(xdist_worker_number % num_rocm_devices)

    # If ROCR_VISIBLE_DEVICES is set, don't also set HIP_VISIBLE_DEVICES
    # (double-filtering can produce HIP_ERROR_NoDevice). Respect the outer setting.
    if os.environ.get("ROCR_VISIBLE_DEVICES"):
      return

    # If present-but-empty, this can hide all GPUs.
    if os.environ.get("HIP_VISIBLE_DEVICES", None) == "":
      del os.environ["HIP_VISIBLE_DEVICES"]

    # HIP layer isolation (ROCm also accepts CUDA_VISIBLE_DEVICES, but we avoid it here).
    os.environ["HIP_VISIBLE_DEVICES"] = assigned

def pytest_configure(config: pytest.Config) -> None:
  """Register custom pytest markers and print attached GPUs to xdist workers."""
    config.addinivalue_line(
      "markers",
      "mosaic_gpu: tests that use Mosaic GPU (skipped on ROCm)",
  )
  
  # Real pytest hook (runs early in main + each xdist worker).
  xdist_worker_name = os.environ.get("PYTEST_XDIST_WORKER", "") or "main"

  # xdist master: print planned mapping (worker stdout is often hidden)
  numproc = int(getattr(getattr(config, "option", None), "numprocesses", 0) or 0)
  if xdist_worker_name == "main" and numproc > 0:
    hip0 = (os.environ.get("HIP_VISIBLE_DEVICES") or "").strip()
    cuda_x = (os.environ.get("JAX_ENABLE_CUDA_XDIST") or "").strip()
    tpu_x = (os.environ.get("JAX_ENABLE_TPU_XDIST") or "").strip()
    rocm_x = (os.environ.get("JAX_ENABLE_ROCM_XDIST") or "").strip()
    if cuda_x:
      try:
        ndev = int(cuda_x)
      except ValueError:
        ndev = 0
      if ndev > 0:
        mapping = ", ".join(f"gw{i}->CUDA_VISIBLE_DEVICES={i % ndev}" for i in range(numproc))
        print(f"[DeviceVisibility] xdist planned mapping: {mapping}", flush=True)
    elif tpu_x:
      mapping = ", ".join(f"gw{i}->TPU_VISIBLE_CHIPS={i}" for i in range(numproc))
      print(f"[DeviceVisibility] xdist planned mapping: {mapping}", flush=True)
    elif rocm_x:
      try:
        ndev = int(rocm_x)
      except ValueError:
        ndev = 0
      if ndev > 0:
        mapping = ", ".join(f"gw{i}->HIP_VISIBLE_DEVICES={i % ndev}" for i in range(numproc))
        print(f"[DeviceVisibility] xdist planned mapping: {mapping}", flush=True)
    elif hip0:
      print(f"[DeviceVisibility] master HIP_VISIBLE_DEVICES={hip0}", flush=True)

  if os.environ.get("JAX_ENABLE_TPU_XDIST", None):
    if xdist_worker_name.startswith("gw"):
      xdist_worker_number = int(xdist_worker_name[len("gw") :])
      os.environ.setdefault("TPU_VISIBLE_CHIPS", str(xdist_worker_number))
      os.environ.setdefault("ALLOW_MULTIPLE_LIBTPU_LOAD", "true")

  elif num_cuda_devices := os.environ.get("JAX_ENABLE_CUDA_XDIST", None):
    if xdist_worker_name.startswith("gw"):
      num_cuda_devices = int(num_cuda_devices)
      xdist_worker_number = int(xdist_worker_name[len("gw") :])
      os.environ.setdefault(
          "CUDA_VISIBLE_DEVICES", str(xdist_worker_number % num_cuda_devices)
      )

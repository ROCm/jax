# Copyright 2025 The JAX Authors.
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

import importlib
import logging
from typing import Any

from .plugin_support import import_from_plugin

logger = logging.getLogger(__name__)

_hip_aiter = import_from_plugin("rocm", "_aiter")

def registrations() -> dict[str, list[tuple[str, Any]]]:
  registrations: dict[str, list[tuple[str, Any]]] = {
      "ROCM": [],
  }
  if _hip_aiter:
    registrations["ROCM"].extend(
        (name, value) for name, value in _hip_aiter.registrations().items()
    )
  else:
    logger.warning(
        "AITer: _aiter module not loaded. FFI handlers will not be registered. "
        "Ensure the jax-rocm plugin wheel is installed with _aiter.so."
    )
  return registrations

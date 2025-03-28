#!/usr/bin/env python3

# Copyright 2024 The JAX Authors.
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

import unittest

import importlib.util
import importlib.machinery


def load_ci_build():
    spec = importlib.util.spec_from_loader(
        "ci_build", importlib.machinery.SourceFileLoader("ci_build", "./ci_build")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ci_build = load_ci_build()


class CIBuildTestCase(unittest.TestCase):
    def test_parse_gpu_targets_spaces(self):
        targets = ["gfx908", "gfx940", "gfx1201"]
        r = ci_build.parse_gpu_targets(" ".join(targets))
        self.assertEqual(r, targets)

    def test_parse_gpu_targets_commas(self):
        targets = ["gfx908", "gfx940", "gfx1201"]
        r = ci_build.parse_gpu_targets(",".join(targets))
        self.assertEqual(r, targets)

    def test_parse_gpu_targets_empty_string(self):
        expected = ci_build.DEFAULT_GPU_DEVICE_TARGETS.split(",")
        r = ci_build.parse_gpu_targets("")
        self.assertEqual(r, expected)

    def test_parse_gpu_targets_whitespace_only(self):
        self.assertRaises(ValueError, ci_build.parse_gpu_targets, " ")

    def test_parse_gpu_targets_invalid_arch(self):
        targets = ["gfx908", "gfx940", "--oops", "/jax"]
        self.assertRaises(ValueError, ci_build.parse_gpu_targets, " ".join(targets))


if __name__ == "__main__":
    unittest.main()

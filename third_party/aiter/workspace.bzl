# Copyright 2026 The JAX Authors.
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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def aiter_mha_whl():
    http_archive(
        name = "aiter_mha_wheel",
        urls = [
            "https://github.com/ROCm/jax/releases/download/AITER-MHA/aiter_mha-0.1.0-py3-none-linux_x86_64.whl",
        ],
        sha256 = "84a5528305c0fa943829f7bbe047a6c5f320273f52b47e7d81ffcca38e68eee6",
        type = "zip",
        strip_prefix = "aiter_mha-0.1.0.data/platlib",
        build_file_content = """\
package(default_visibility = ["//visibility:public"])
exports_files(["libmha_fwd.so", "libmha_bwd.so"])
""",
    )
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
        sha256 = "943e8736ded39244acab96c0c4e3901aa0cb166fdbcd1e0bee178799e21868c2",
        type = "zip",
        strip_prefix = "aiter_mha-0.1.0.data",
        build_file_content = """\
package(default_visibility = ["//visibility:public"])

exports_files([
    "libs/libmha_fwd.so",
    "libs/libmha_bwd.so",
])

cc_library(
    name = "aiter_headers",
    hdrs = [
        "include/aiter_hip_common.h",
        "include/aiter_logger.h",
        "include/mha_bwd.h",
        "include/mha_fwd.h",
    ],
    strip_include_prefix = "include",
    include_prefix = "aiter",
)
""",
    )
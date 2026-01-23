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

"""Repository rule to parse plugin wheel dependencies from environment variable.

This module provides a repository rule that reads the PLUGIN_WHEEL_DEPS
environment variable and generates a .bzl file containing a list of targets
that can be used as dependencies.

The environment variable should contain a comma-separated list of targets, e.g.:
    PLUGIN_WHEEL_DEPS=@jax_rocm_plugin//:plugin.whl,@jax_rocm_plugin//:pjrt.whl

Usage:
    # In WORKSPACE:
    load("//third_party/plugins:workspace.bzl", "plugin_wheel_deps_repository")
    plugin_wheel_deps_repository(name = "plugin_wheel_deps")

    # In BUILD files:
    load("@plugin_wheel_deps//:deps.bzl", "PLUGIN_WHEEL_DEPS")
    py_library(
        name = "my_target",
        deps = PLUGIN_WHEEL_DEPS,
    )
"""

def _plugin_wheel_deps_repository_impl(repository_ctx):
    """Implementation of the plugin_wheel_deps_repository rule.

    Reads the PLUGIN_WHEEL_DEPS environment variable and generates a deps.bzl
    file containing a list of targets.

    Args:
        repository_ctx: The repository context.
    """
    env_var_name = repository_ctx.attr.env_var
    deps_env = repository_ctx.os.environ.get(env_var_name, "")

    # Parse the comma-separated list of targets
    deps_list = []
    if deps_env:
        for dep in deps_env.split(","):
            dep = dep.strip()
            if dep:
                deps_list.append(dep)

    # Generate the deps.bzl file with the list of targets
    deps_content = """\
# Auto-generated file. Do not edit.
# Generated from environment variable: {env_var}

# List of plugin wheel dependency targets.
# These targets can be used as dependencies in BUILD files.
PLUGIN_WHEEL_DEPS = [
{deps}
]
""".format(
        env_var = env_var_name,
        deps = "\n".join(['    "{}",'.format(dep) for dep in deps_list]),
    )

    repository_ctx.file("deps.bzl", deps_content)
    repository_ctx.file("BUILD.bazel", "# Auto-generated BUILD file\n")

plugin_wheel_deps_repository = repository_rule(
    implementation = _plugin_wheel_deps_repository_impl,
    attrs = {
        "env_var": attr.string(
            default = "PLUGIN_WHEEL_DEPS",
            doc = "The name of the environment variable containing the comma-separated list of targets.",
        ),
    },
    environ = ["PLUGIN_WHEEL_DEPS"],
    doc = """Repository rule to parse plugin wheel dependencies from an environment variable.

Reads a comma-separated list of Bazel targets from the specified environment
variable (default: PLUGIN_WHEEL_DEPS) and generates a deps.bzl file that
exports a PLUGIN_WHEEL_DEPS list containing these targets.

Example:
    # Set the environment variable:
    export PLUGIN_WHEEL_DEPS="@jax_rocm_plugin//:plugin.whl,@jax_rocm_plugin//:pjrt.whl"

    # In WORKSPACE:
    load("//third_party/plugins:workspace.bzl", "plugin_wheel_deps_repository")
    plugin_wheel_deps_repository(name = "plugin_wheel_deps")

    # In BUILD files:
    load("@plugin_wheel_deps//:deps.bzl", "PLUGIN_WHEEL_DEPS")
    py_library(
        name = "my_target",
        deps = PLUGIN_WHEEL_DEPS,
    )
""",
)

def repo(name = "plugin_wheel_deps"):
    """Convenience function to create the plugin wheel deps repository.

    Args:
        name: The name of the repository (default: "plugin_wheel_deps").
    """
    plugin_wheel_deps_repository(name = name)

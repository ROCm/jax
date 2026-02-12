"""Wrapper macros for ROCm wheel RPATH configuration.

When building wheels with rocm_path_type=link_only, Bazel sandbox rpaths
are stripped and replaced with wheel-relative paths so that .so files
find ROCm libraries at install time.

Usage in BUILD files:

    load("//jaxlib/rocm:rpath.bzl", "rocm_nanobind_extension")

    rocm_nanobind_extension(
        name = "_my_ext",
        srcs = ["my_ext.cc"],
        ...
    )
"""

load("//jaxlib:jax.bzl", "nanobind_extension")

_ROCM_LINK_ONLY = "@local_config_rocm//rocm:link_only"

_WHEEL_RPATHS = [
    "-Wl,-rpath,$$ORIGIN/../rocm/lib",
    "-Wl,-rpath,$$ORIGIN/../../rocm/lib",
    "-Wl,-rpath,/opt/rocm/lib",
]

def _wheel_features():
    return select({
        _ROCM_LINK_ONLY: ["no_solib_rpaths"],
        "//conditions:default": [],
    })

def _wheel_linkopts():
    return select({
        _ROCM_LINK_ONLY: _WHEEL_RPATHS,
        "//conditions:default": [],
    })

def rocm_nanobind_extension(name, features = [], linkopts = [], **kwargs):
    """nanobind_extension that automatically strips solib rpaths and embeds wheel RPATHs.

    When built with --@local_config_rocm//rocm:rocm_path_type=link_only,
    the no_solib_rpaths feature is enabled and wheel-specific RPATHs are
    added. Otherwise the target behaves identically to nanobind_extension.

    Args:
        name: Target name.
        features: Additional features (rpath features are appended automatically).
        linkopts: Additional linkopts (wheel RPATHs are appended automatically).
        **kwargs: Passed through to nanobind_extension.
    """
    nanobind_extension(
        name = name,
        features = features + _wheel_features(),
        linkopts = linkopts + _wheel_linkopts(),
        **kwargs
    )

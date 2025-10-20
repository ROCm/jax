load("@rules_cc//cc:defs.bzl", "cc_toolchain", "cc_toolchain_suite")
#load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])

platform(
    name = "linux_x64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:48515bbef9780ce32bdadd6e236d344217009fec4ed47a7809f41707ef8b8551",
        "OSFamily": "Linux",
        "Pool": "linux_x64_large",
    },
)

platform(
    name = "linux_x64_gpu",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
        "@bazel_tools//tools/cpp:clang",
    ],
    exec_properties = {
        "container-image": "docker://938540721288.dkr.ecr.us-east-1.amazonaws.com/amd/rbe@sha256:48515bbef9780ce32bdadd6e236d344217009fec4ed47a7809f41707ef8b8551",
        "OSFamily": "Linux",
        "Pool": "linux_x64_gpu",
    },
)


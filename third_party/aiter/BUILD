package(default_visibility = ["//visibility:public"])

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

cc_import(
    name = "mha_fwd_so",
    shared_library = "libmha_fwd.so",
    deps = [":aiter_headers"],
)

cc_import(
    name = "mha_bwd_so",
    shared_library = "libmha_bwd.so",
    deps = [ ":aiter_headers"],
)

cc_library(
    name = "aiter_mha",
    deps = [
        ":mha_fwd_so",
        ":mha_bwd_so",
    ],
)

filegroup(
    name = "aiter_runtime_sos",
    srcs = [
        "libmha_fwd.so",
        "libmha_bwd.so",
    ],
)
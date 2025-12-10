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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("@rules_python//python:defs.bzl", "py_library")
load("//jaxlib:jax.bzl", "if_oss", "nanobind_extension")


cc_library(
    name = "custom_call",
    srcs = ["custom_call.cc"],
    visibility = ["//visibility:public"], # to let HW abstracted custom_call on upper level see it.
    linkopts = if_oss(
        [],  # We use a version script to ensure symbol visibility in the OSS build.
        [
            "-Wl,--export-dynamic-symbol='MosaicGpuCompile'",
            "-Wl,--export-dynamic-symbol='MosaicGpuUnload'",
        ],
    ),
    deps = [
        #":dump",
        #":library_paths",  # buildcleaner: keep
        #":nvshmem",
        #":passes",
        #":serde",
        #":target",
        "//jaxlib/rocm:hip_vendor",
        "//jaxlib/mosaic/dialect/gpu:mosaic_gpu",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:ArithDialect",
        "@llvm-project//mlir:ArithToLLVM",
        "@llvm-project//mlir:ArithTransforms",
        "@llvm-project//mlir:BuiltinToLLVMIRTranslation",
        "@llvm-project//mlir:ComplexToLLVM",
        "@llvm-project//mlir:ControlFlowDialect",
        "@llvm-project//mlir:ControlFlowToLLVM",
        "@llvm-project//mlir:ConversionPasses",
        "@llvm-project//mlir:ExecutionEngine",
        "@llvm-project//mlir:ExecutionEngineUtils",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:FuncToLLVM",
        "@llvm-project//mlir:GPUDialect",
        "@llvm-project//mlir:GPUToLLVMIRTranslation",
        "@llvm-project//mlir:GPUTransforms",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:IndexToLLVM",
        "@llvm-project//mlir:LLVMDialect",
        "@llvm-project//mlir:LLVMIRTransforms",
        "@llvm-project//mlir:LLVMToLLVMIRTranslation",
        "@llvm-project//mlir:MathDialect",
        "@llvm-project//mlir:MathToLLVM",
        "@llvm-project//mlir:MemRefDialect",
        "@llvm-project//mlir:MemRefToLLVM",
        "@llvm-project//mlir:MemRefTransforms",
        #"@llvm-project//mlir:NVGPUDialect",
        #"@llvm-project//mlir:NVVMDialect",
        #"@llvm-project//mlir:NVVMTarget",
        #"@llvm-project//mlir:NVVMToLLVM",
        #"@llvm-project//mlir:NVVMToLLVMIRTranslation",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:SCFDialect",
        "@llvm-project//mlir:Support",
        "@llvm-project//mlir:Transforms",
        "@llvm-project//mlir:UBToLLVM",
        "@llvm-project//mlir:VectorDialect",
        "@llvm-project//mlir:VectorToLLVM",
        #"@local_config_cuda//cuda:cuda_headers",
        "@tsl//tsl/platform:path",
        "@tsl//tsl/profiler/lib:traceme",
        "@xla//xla:executable_run_options",
        "@xla//xla/ffi",
        "@xla//xla/ffi:ffi_api",
        "@xla//xla/service:custom_call_status",
        "@xla//xla/service:custom_call_target_registry",
        #"@xla//xla/service/gpu/llvm_gpu_backend:nvptx_libdevice_path",
        #"@xla//xla/stream_executor/cuda:assemble_compilation_provider",
        #"@xla//xla/stream_executor/cuda:compilation_provider",
        #"@xla//xla/stream_executor/cuda:compilation_provider_options",
        #"@xla//xla/stream_executor/cuda:cuda_compute_capability",
        #"@xla//xla/stream_executor/cuda:ptx_compiler_support",
        "@xla//xla/tsl/platform:statusor",
    ],
    alwayslink = True,
)

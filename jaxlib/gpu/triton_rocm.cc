#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/vendor.h"
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep
#include "xla/service/custom_call_status.h"

#define HIP_RETURN_IF_ERROR(expr) JAX_RETURN_IF_ERROR(JAX_AS_STATUS(expr))

namespace py = pybind11;

namespace jax::JAX_GPU_NAMESPACE {

// TODO(cjfj): Move this to `gpu_kernel_helpers`?
// Used via JAX_AS_STATUS(expr) macro.
absl::Status AsStatus(hipError_t error, const char* file, std::int64_t line,
                      const char* expr) {
  if (ABSL_PREDICT_TRUE(error == HIP_SUCCESS)) {
    return absl::OkStatus();
  }

  const char* str;
  //CHECK_EQ(cuGetErrorName(error, &str), HIP_SUCCESS);
  CHECK_EQ(error, HIP_SUCCESS);
  return absl::InternalError(
      absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr, str));
}

}  // namespace jax::JAX_GPU_NAMESPACE

namespace jax_triton {
namespace {

constexpr uint32_t kNumThreadsPerWarp = 64;

struct ROCmModuleDeleter {
  void operator()(hipModule_t module) { hipModuleUnload(module); }
};

using OwnedROCmmodule =
    std::unique_ptr<std::remove_pointer_t<hipModule_t>, ROCmModuleDeleter>;

class TritonKernel {
 public:
  TritonKernel(std::string module_image, std::string kernel_name,
               uint32_t num_warps, uint32_t shared_mem_bytes)
      : module_image_(std::move(module_image)),
        kernel_name_(std::move(kernel_name)),
        block_dim_x_(num_warps * kNumThreadsPerWarp),
        shared_mem_bytes_(shared_mem_bytes) {}

  absl::Status Launch(hipStream_t stream, uint32_t grid[3], void** params) {
    //hipCtx_t context;
    //HIP_RETURN_IF_ERROR(cuStreamGetCtx(stream, &context));
    //absl::StatusOr<hipFunction_t> kernel = GetFunctionForContext(context);
    absl::StatusOr<hipFunction_t> kernel = GetFunctionForStream(stream);
    JAX_RETURN_IF_ERROR(kernel.status());
    return JAX_AS_STATUS(hipModuleLaunchKernel(
        *kernel, grid[0], grid[1], grid[2], block_dim_x_,
        /*blockDimY=*/1, /*blockDimZ=*/1, shared_mem_bytes_, stream, params,
        /*extra=*/nullptr));
  }

 private:
  absl::StatusOr<hipFunction_t> GetFunctionForStream(hipStream_t stream) {
    absl::MutexLock lock(&mutex_);
    /*auto it = functions_.find(context);
    if (it != functions_.end()) {
      return it->second;
    }*/

    //HIP_RETURN_IF_ERROR(hipCtxPushCurrent(context));
    //absl::Cleanup ctx_restorer = [] { hipCtxPopCurrent(nullptr); };

    hipModule_t module;
    HIP_RETURN_IF_ERROR(hipModuleLoadData(&module, module_image_.c_str()));
    modules_.push_back(OwnedROCmmodule(module, ROCmModuleDeleter()));

    hipFunction_t function;
    HIP_RETURN_IF_ERROR(
        hipModuleGetFunction(&function, module, kernel_name_.c_str()));
    //auto [_, success] = functions_.insert({context, function});
    //CHECK(success);

    // The maximum permitted static shared memory allocation in CUDA is 48kB,
    // but we can expose more to the kernel using dynamic shared memory.
    constexpr int kMaxStaticSharedMemBytes = 49152;
    if (shared_mem_bytes_ <= kMaxStaticSharedMemBytes) {
      return function;
    }

    // Set up dynamic shared memory.
    hipDevice_t device;
    HIP_RETURN_IF_ERROR(hipStreamGetDevice(stream, &device));

    int shared_optin;
    HIP_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &shared_optin, hipDeviceAttributeSharedMemPerBlockOptin,
        device));

    if (shared_mem_bytes_ > shared_optin) {
      return absl::InvalidArgumentError(
          "Shared memory requested exceeds device resources.");
    }

    if (shared_optin > kMaxStaticSharedMemBytes) {
      HIP_RETURN_IF_ERROR(
          hipFuncSetCacheConfig(function, hipFuncCachePreferShared));
      int shared_total;
      HIP_RETURN_IF_ERROR(hipDeviceGetAttribute(
          &shared_total,
          hipDeviceAttributeMaxBlocksPerMultiProcessor , device));
      int shared_static;
      HIP_RETURN_IF_ERROR(hipFuncGetAttribute(
          &shared_static, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, function));
      HIP_RETURN_IF_ERROR(hipFuncSetAttribute(
          function, hipFuncAttributeMaxDynamicSharedMemorySize,
          shared_optin - shared_static));
    }
    return function;
  }

  std::string module_image_;
  std::string kernel_name_;
  uint32_t block_dim_x_;
  uint32_t shared_mem_bytes_;

  absl::Mutex mutex_;
  std::vector<OwnedROCmmodule> modules_ ABSL_GUARDED_BY(mutex_);
  //absl::flat_hash_map<hipCtx_t, hipFunction_t> functions_ ABSL_GUARDED_BY(mutex_);
};

struct TritonKernelCallBase {
  virtual ~TritonKernelCallBase() = default;
  virtual absl::Status Launch(hipStream_t stream, void** buffers) = 0;
};

class TritonKernelCall: public TritonKernelCallBase {
  public:
    struct ArrayParameter {
      size_t bytes_to_zero;
      bool ptr_must_be_divisible_by_16;
    };

    //Parameters can be either to either arrays or scalars (encoded as uint64).
    using Parameter = std::variant<ArrayParameter, uint64_t>;

    TritonKernelCall(TritonKernel& kernel, uint32_t grid_0, uint32_t grid_1, 
                  uint32_t grid_2, std::vector<Parameter> parameters)
                  : kernel_(kernel),
                    grid_{grid_0, grid_1, grid_2},
                    parameters_(std::move(parameters)) {}
    
    absl::Status Launch(hipStream_t stream, void** buffers) override final {
      std::vector<void*> params;
      params.reserve(parameters_.size());
      for (size_t i = 0; i < parameters_.size(); ++i) {
        const Parameter& param = parameters_[i];
        if(std::holds_alternative<ArrayParameter>(param)) {
          const ArrayParameter& array = std::get<ArrayParameter>(param);
          void*& ptr = *(buffers++);
          auto cu_ptr = reinterpret_cast<hipDeviceptr_t>(ptr);

          if (array.ptr_must_be_divisible_by_16 && (uint64_t(cu_ptr) % 16 != 0)) {
            return absl::InvalidArgumentError(absl::StrFormat(
              "Parameter %zu (%p) is not divisible by 16.", i, ptr));
          }

          if(array.bytes_to_zero > 0) {
            HIP_RETURN_IF_ERROR(
              hipMemsetD8Async(cu_ptr, 0, array.bytes_to_zero, stream));
          }
          params.push_back(&ptr);
        } else {
          params.push_back(const_cast<uint64_t*>(&std::get<uint64_t>(param)));
        }
      }

      return kernel_.Launch(stream, grid_, params.data());
    }
  private:
    TritonKernel& kernel_;
    uint32_t grid_[3];
    std::vector<Parameter> parameters_;

};

/*
class TritonAutotunedKernelCall: public TritonKernelCallBase {
  public:
    struct Config {
      py::object kernel_call;
      std::string description;
    };

    TritonAutotunedKernelCall(
      std::string name, std::vector<Config> configs,
      std::vector<std::tuple<size_t, size_t, size_t>> input_output_aliases)
      : name_(std::move(name)),
        configs_(std::move(configs)),
        input_output_aliases_(std::move(input_output_aliases)) {}
    

    private:
      static constexpr float kBenchmarkTimeMillis = 10;

      absl::Status Autotune(hipStream_t stream, void** buffers) {
        //Ensure a valid context for driver calls that don't take the stream.
        
      }
}*/

template <typename CppT, typename PyT>
uint64_t EncodeKernelParameterAs(PyT value) {
  static_assert(sizeof(CppT) <= sizeof(uint64_t));
  union {
    CppT value;
    uint64_t bits;
  } encoded;
  encoded.bits = 0;
  encoded.value = CppT(value);
  return encoded.bits;
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::int_ value, 
                                              std::string_view dtype) {
  if ((dtype == "i1") || (dtype == "i8")) {
    return EncodeKernelParameterAs<int8_t>(value);
  } else if(dtype == "u8") {
    return EncodeKernelParameterAs<uint8_t>(value);
  } else if(dtype == "i16") {
    return EncodeKernelParameterAs<int16_t>(value);
  } else if(dtype == "u16") {
    return EncodeKernelParameterAs<uint16_t>(value);
  } else if(dtype == "i32") {
    return EncodeKernelParameterAs<int32_t>(value);
  } else if(dtype == "u32") {
    return EncodeKernelParameterAs<int32_t>(value);
  } else if(dtype == "i64") {
    return EncodeKernelParameterAs<int64_t>(value);
  } else if(dtype == "u64") {
    return EncodeKernelParameterAs<uint64_t>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype:") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::float_ value,
                                               std::string_view dtype) {
  if (dtype == "fp32") {
    return EncodeKernelParameterAs<float>(value);
  } else if (dtype == "fp64") {
    return EncodeKernelParameterAs<double>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

absl::StatusOr<uint64_t> EncodeKernelParameter(py::bool_ value,
                                               std::string_view dtype) {
  if ((dtype == "int1") || (dtype == "B")) {
    return EncodeKernelParameterAs<bool>(value);
  } else {
    return absl::InvalidArgumentError(std::string("unknown dtype: ") +
                                      dtype.data());
  }
}

}  // namespace


void LaunchTritonKernel(hipStream_t stream, void** buffers, const char* opaque,
                        size_t opaque_len, XlaCustomCallStatus* status) {
    CHECK_EQ(opaque_len, sizeof(TritonKernelCallBase*));
    TritonKernelCallBase* kernel_call;
    std::memcpy(&kernel_call, opaque, sizeof(TritonKernelCallBase*));
    absl::Status result = kernel_call->Launch(stream, buffers);
    if (!result.ok()) {
      absl::string_view msg = result.message();
      XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
    }
}

PYBIND11_MODULE(_triton, m) {
  py::class_<TritonKernel>(m, "TritonKernel")
      .def(py::init<std::string, std::string, uint32_t, uint32_t>());

  
  py::class_<TritonKernelCall>(m, "TritonKernelCall")
      .def(py::init<TritonKernel&, uint32_t, uint32_t, uint32_t,
                    std::vector<TritonKernelCall::Parameter>>(),
           py::keep_alive<1, 2>())  // Ensure that the kernel lives long enough.
      .def_property_readonly("descriptor", [](TritonKernelCall& kernel_call) {
        union {
          TritonKernelCall* ptr;
          char bytes[sizeof(TritonKernelCall*)];
        } descriptor;
        descriptor.ptr = &kernel_call;
        return py::bytes(descriptor.bytes, sizeof(TritonKernelCall*));
      });

  py::class_<TritonKernelCall::ArrayParameter>(m, "TritonArrayParameter");

  m.def("get_custom_call", [] {
    return py::capsule(reinterpret_cast<void*>(&LaunchTritonKernel),
                       "xla._CUSTOM_CALL_TARGET");
  });

  m.def("create_array_parameter",
        [](size_t bytes_to_zero, bool ptr_must_be_divisible_by_16) {
          return TritonKernelCall::ArrayParameter{bytes_to_zero,
                                                  ptr_must_be_divisible_by_16};
        });
  m.def("create_scalar_parameter",
        py::overload_cast<py::int_, std::string_view>(&EncodeKernelParameter));
  m.def(
      "create_scalar_parameter",
      py::overload_cast<py::float_, std::string_view>(&EncodeKernelParameter));
  m.def("create_scalar_parameter",
        py::overload_cast<py::bool_, std::string_view>(&EncodeKernelParameter));
  m.def("get_compute_capability", [](int device) -> absl::StatusOr<int> {
    int major, minor;
    HIP_RETURN_IF_ERROR(hipInit(device));
    HIP_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &major, hipDeviceAttributeComputeCapabilityMajor , device));
    HIP_RETURN_IF_ERROR(hipDeviceGetAttribute(
        &minor, hipDeviceAttributeComputeCapabilityMajor , device));
    return major * 10 + minor;
  });
}

}  // namespace jax_triton

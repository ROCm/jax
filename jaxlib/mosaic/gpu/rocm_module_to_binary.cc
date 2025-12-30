/* Copyright 2025 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/mosaic/gpu/rocm_module_to_binary.h"

// #include <cassert>
#include <memory>
// #include <optional>
#include <string>
#include <utility>
// #include <vector>
//
// #include "absl/base/call_once.h"
#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

// #include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
// #include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "jaxlib/mosaic/pass_boilerplate.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "xla/service/gpu/llvm_gpu_backend/load_ir_module.h"

namespace mosaic {
namespace gpu {
namespace {

using ::llvm::LogicalResult;
using ::llvm::SmallVector;
using ::mlir::Attribute;
using ::mlir::gpu::GPUModuleOp;

class ModuleToBinary : public mlir::LLVM::ModuleToObject {
public:
  ModuleToBinary(gpu::GPUModuleOp gpu_module,
                 ::mlir::ROCDL::ROCDLTargetAttr target /*,
                 std::vector<std::string> libraries_to_link*/)
      : ModuleToObject(*gpu_module,
                       target.getTriple(), target.getChip(),
                       target.getFeatures(), target.getO()
                       )
        // , libraries_to_link_(std::move(libraries_to_link))
         {};

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvm_module) override {
    std::optional<llvm::TargetMachine *> machine = getOrCreateTargetMachine();
    if (!machine) {
      getOperation().emitError() << "Target Machine unavailable for triple "
                                 << triple << ", can't optimize with LLVM\n";
      return std::nullopt;
    }
    std::optional<std::string> ptx = translateToISA(llvm_module, **machine);
    if (!ptx) {
      getOperation().emitError() << "Failed translating the module to PTX.";
      return std::nullopt;
    }

    return SmallVector<char, 0>(ptx->begin(), ptx->end());
  }

  // Loads the bitcode files in `libraries_to_link_`.
  /*std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
  loadBitcodeFiles(llvm::Module &llvm_module) override {
    llvm::LLVMContext &ctx = llvm_module.getContext();
    llvm::SMDiagnostic err;
    SmallVector<std::unique_ptr<llvm::Module>> loaded_modules;
    loaded_modules.reserve(libraries_to_link_.size());
    for (const std::string &library_path : libraries_to_link_) {
      std::unique_ptr<llvm::Module> library_module =
          xla::gpu::LoadIRModule(library_path, &ctx);
      if (!library_module) {
        getOperation().emitError()
            << "Failed loading file from " << library_path
            << ", error: " << err.getMessage();
        return std::nullopt;
      }
      loaded_modules.push_back(std::move(library_module));
    }
    return loaded_modules;
  }*/
/*
private:
  std::vector<std::string> libraries_to_link_;*/
};

LogicalResult
LowerGpuModuleToBinary(GPUModuleOp gpu_module /*,
                       const std::vector<std::string> &libraries_to_link*/) {
  // EnsureLLVMNVPTXTargetIsRegistered();
  mlir::gpu::OffloadingLLVMTranslationAttrInterface handler(nullptr);
  mlir::OpBuilder builder(gpu_module->getContext());
  SmallVector<Attribute> objects;

  // Fail if there are no target attributes
  if (gpu_module.getTargetsAttr().size() != 1) {
    return gpu_module.emitError(
               "Expected exactly one target attribute, but got ")
           << gpu_module.getTargetsAttr().size();
  }

  /*auto target_attr = llvm::dyn_cast<mlir::NVVM::NVVMTargetAttr>(
      gpu_module.getTargetsAttr()[0]);
  if (!target_attr) {
    return gpu_module.emitError(
        "Target attribute is not of type NVVMTargetAttr");
  }
  ModuleToBinary serializer(gpu_module, target_attr, libraries_to_link);*/

  auto target_attr = llvm::dyn_cast<mlir::ROCDL::ROCDLTargetAttr>(
      gpu_module.getTargetsAttr()[0]);
  if (!target_attr) {
    return gpu_module.emitError(
        "Target attribute is not of type ROCDLTargetAttr");
  }

  /*auto target_attr = mlir::ROCDL::ROCDLTargetAttr::get(
      gpu_module->getContext(), // MLIRContext*
      3,                        // int (e.g., 2 or 3)
      "amdgcn-amd-amdhsa",      // triple (StringRef)
      "gfx942",                 // chip (StringRef)
      "",                       // features (StringRef)
      "600"                     // abiVersion (StringRef)
  );*/

  ModuleToBinary serializer(gpu_module, target_attr /*, libraries_to_link*/);

  std::optional<SmallVector<char, 0>> binary = serializer.run();
  if (!binary) {
    gpu_module.emitError("An error happened while serializing the module.");
    return mlir::failure();
  }

  SmallVector<mlir::NamedAttribute> properties{
      builder.getNamedAttr("O", builder.getI32IntegerAttr(target_attr.getO()))};

  Attribute object = builder.getAttr<mlir::gpu::ObjectAttr>(
      target_attr,
      // mlir::gpu::CompilationTarget::Assembly,
      mlir::gpu::CompilationTarget::Binary,
      builder.getStringAttr(llvm::StringRef(binary->data(), binary->size())),
      builder.getDictionaryAttr(properties), /*kernels=*/nullptr);

  if (!object) {
    gpu_module.emitError("An error happened while creating the object.");
    return mlir::failure();
  }

  builder.setInsertionPointAfter(gpu_module);
  mlir::gpu::BinaryOp::create(
      builder, gpu_module.getLoc(), gpu_module.getName(), /*handler=*/nullptr,
      builder.getArrayAttr(SmallVector<Attribute>{object}));
  gpu_module->erase();
  return mlir::success();
}

class RocmModuleToBinaryPass
    : public jaxlib::mlir::Pass<RocmModuleToBinaryPass, mlir::ModuleOp> {
  using BaseClass = jaxlib::mlir::Pass<RocmModuleToBinaryPass, mlir::ModuleOp>;

public:
  RocmModuleToBinaryPass() = default;

  // TODO(Arech): CUDA has a weird implementation of a copy constructor of
  // respective GpuModuleToAssemblyPass: it's just {}, skipping calling the base
  // class c/constructor and not copying libraries_to_link_. libraries_to_link_
  // is however problematic since one of its base classes prohibits copying, so
  // the fix here isn't obvious.
  // So how a correct copy of this class is expected to occur with such an
  // implemnentation?
  RocmModuleToBinaryPass(const RocmModuleToBinaryPass &o) {};

  static constexpr llvm::StringLiteral kArgumentName =
      "mosaic-rocm-module-to-binary";
  static constexpr llvm::StringLiteral kPassName = "RocmModuleToBinaryPass";

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::gpu::GPUModuleOp gpu_module) {
      if (mlir::failed(
              LowerGpuModuleToBinary(gpu_module /*, libraries_to_link_*/))) {
        gpu_module.emitError("Failed to lower GPU module to binary.");
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });
  }

  /*
private:
  ListOption<std::string> libraries_to_link_{
      *this, "libraries-to-link",
      llvm::cl::desc("A comma-separated list of bitcode files to link into the "
                     "resulting assembly.")};*/
};

} // namespace

void registerRocmModuleToBinaryPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<RocmModuleToBinaryPass>();
  });
}

} // namespace gpu
} // namespace mosaic

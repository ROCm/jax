cmake_minimum_required(VERSION 3.15...3.30)

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX HIP)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_FFI_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "XLA FFI include directory: ${XLA_FFI_INCLUDE_DIR}")

find_package(hip REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_library(_hip_examples SHARED
  src/jax_ffi_example/hip_examples.cu
)

set_source_files_properties(
  src/jax_ffi_example/hip_examples.cu
  PROPERTIES LANGUAGE HIP
)

set_target_properties(_hip_examples PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  LINKER_LANGUAGE HIP
)

target_include_directories(_hip_examples PRIVATE
  ${XLA_FFI_INCLUDE_DIR}
)

target_link_libraries(_hip_examples PRIVATE
  hip::host
)

target_compile_options(_hip_examples PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type-c-linkage>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>
)

install(TARGETS _hip_examples
  LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME}
)
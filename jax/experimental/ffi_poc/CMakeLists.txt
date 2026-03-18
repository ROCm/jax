cmake_minimum_required(VERSION 3.15...3.30)

project(${SKBUILD_PROJECT_NAME} LANGUAGES HIP CXX)

message("SKBUILD_PROJECT_NAME=${SKBUILD_PROJECT_NAME}")


if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
else()
    set(ROCM_PATH "/opt/rocm")
endif()

list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
list(APPEND CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip")

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(hip REQUIRED)
find_package(nanobind CONFIG REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
#find_package(hip REQUIRED)
#find_package(nanobind CONFIG REQUIRED)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)



execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
          "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "XLA include directory: ${XLA_DIR}")


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Add subdirectories (Order matters if one depends on another)
add_subdirectory(kernels)
add_subdirectory(src/shared_obj_handler)
add_subdirectory(src/jax_ffi_poc)


install(TARGETS _hip_simple_kernel LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
install(TARGETS _hip_ffi_poc LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})


cmake_minimum_required(VERSION 3.15...3.30)

add_library(_hip_ffi_poc SHARED  "hip_ffi.cu")

set_source_files_properties(hip_ffi.cu PROPERTIES LANGUAGE HIP)

set_target_properties(_hip_ffi_poc PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  LINKER_LANGUAGE HIP
 )

target_include_directories(_hip_ffi_poc PRIVATE
  ${XLA_DIR}
  ${CMAKE_SOURCE_DIR}/kernels
  ${CMAKE_SOURCE_DIR}/src/shared_obj_handler
)

target_compile_options(_hip_ffi_poc PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type-c-linkage>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>
)


target_link_libraries(_hip_ffi_poc PRIVATE
  hip::device
  #_hip_simple_kernel
  lib_handler
  dl
)


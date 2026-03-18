cmake_minimum_required(VERSION 3.15...3.30)

# Specify target architectures (replace with your specific GPU, e.g., gfx906, gfx1032, or "70;75" for NVIDIA)
#set(CMAKE_HIP_ARCHITECTURES "gfx950;gfx942") # Example for AMD


# Add an shared obj target with your source files (.cpp, .hip, etc.)
add_library(_hip_simple_kernel SHARED simple_hip_kernel.cu)
set_source_files_properties(simple_hip_kernel.cu PROPERTIES LANGUAGE HIP)

set_target_properties(_hip_simple_kernel PROPERTIES 
                                        POSITION_INDEPENDENT_CODE ON
                                        HIP_STANDARD 17
                                        HIP_STANDARD_REQUIRED ON
                                        LINKER_LANGUAGE HIP
)

target_include_directories(_hip_simple_kernel PRIVATE  ${XLA_DIR})

# Link the necessary HIP library
target_link_libraries(_hip_simple_kernel PRIVATE hip::device dl)

target_compile_options(_hip_simple_kernel PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-return-type-c-linkage>
  $<$<COMPILE_LANGUAGE:CXX>:-Wno-return-type-c-linkage>
)

# Export headers
target_include_directories(_hip_simple_kernel PUBLIC include/)

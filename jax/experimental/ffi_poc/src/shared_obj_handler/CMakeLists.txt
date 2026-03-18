# Define the static library
add_library(lib_handler STATIC shared_libs.cc)

set_source_files_properties(shared_libs.cc PROPERTIES LANGUAGE HIP)

set_target_properties(lib_handler PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  HIP_STANDARD 17
  HIP_STANDARD_REQUIRED ON
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  LINKER_LANGUAGE HIP
)

# Export include directory so others can find your .h files
target_include_directories(lib_handler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

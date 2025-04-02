set(attention_files
    attention.cu
)

set (ATTENTION_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/flash_attention
    
)

include_directories(
    ${ATTENTION_HEADER_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/flash_attention/lib)

message("-- Assign attention include directories (include_directories=${ATTENTION_HEADER_DIR})")
message("-- in attention cuda Assign arch (arch=${CMAKE_CUDA_ARCHITECTURES})")

add_library(attention SHARED ${attention_files})
# Request that particles be built with --std=c++14
# As this is a public compile feature anything that links to particles
# will also build with -std=c++14
target_compile_features(attention PUBLIC cxx_std_14)
target_compile_options(attention PUBLIC "-gencode=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")
set_target_properties(attention PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(attention PUBLIC -lcudart -lcublas)


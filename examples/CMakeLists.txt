set (sample_attention_baseline_files
    attention_baseline.cu
)

set (sample_minmal_flash_attention_v1_files
    minmal_flash_attention.cu
)

set (sample_flash_attn_v1_tiling
    flash_attn_v1_tiling.cu
)

set (sample_flash_attn_v1_tiling_warp_parallel
    flash_attn_v1_tiling_warp_parallel.cu
)

set (sample_flash_attn_v1_tiling_D
    flash_attn_v1_tiling_D.cu
)
set (sample_flash_attn_v2_tiling
    flash_attn_v2_tiling.cu
)
set (sample_flash_attn_v2_mma
    flash_attn_v2_mma.cu
)
set (sample_flash_attn_v2_mma_swizzle
    flash_attn_v2_mma_swizzle.cu
)

set (SAMPLES_HEADER_DIR
    ${PROJECT_SOURCE_DIR}/flash_attention
)

include_directories(
    ${SAMPLES_HEADER_DIR}
)

link_directories(
    ${PROJECT_SOURCE_DIR}/flash_attention/lib
)

message("-- Assign include directories (include_directories=${SAMPLES_HEADER_DIR})")

add_executable(attention_baseline ${sample_attention_baseline_files})
target_compile_features(attention_baseline PUBLIC cxx_std_14)
target_link_libraries(attention_baseline PUBLIC attention)

add_executable(minmal_flash_attention ${sample_minmal_flash_attention_v1_files})
target_compile_features(minmal_flash_attention PUBLIC cxx_std_14)
target_link_libraries(minmal_flash_attention PUBLIC attention)

add_executable(flash_attn_v1_tiling ${sample_flash_attn_v1_tiling})
target_compile_features(flash_attn_v1_tiling PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v1_tiling PUBLIC attention)

add_executable(flash_attn_v1_tiling_warp_parallel ${sample_flash_attn_v1_tiling_warp_parallel})
target_compile_features(flash_attn_v1_tiling_warp_parallel PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v1_tiling_warp_parallel PUBLIC attention)

add_executable(flash_attn_v1_tiling_D ${sample_flash_attn_v1_tiling_D})
target_compile_features(flash_attn_v1_tiling_D PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v1_tiling_D PUBLIC attention)

add_executable(flash_attn_v2_tiling ${sample_flash_attn_v2_tiling})
target_compile_features(flash_attn_v2_tiling PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v2_tiling PUBLIC attention)

add_executable(flash_attn_v2_mma ${sample_flash_attn_v2_mma})
target_compile_features(flash_attn_v2_mma PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v2_mma PUBLIC attention)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash_attn_v2_mma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    target_compile_options(flash_attn_v2_mma PRIVATE -lineinfo)
endif()

add_executable(flash_attn_v2_mma_swizzle ${sample_flash_attn_v2_mma_swizzle})
target_compile_features(flash_attn_v2_mma_swizzle PUBLIC cxx_std_14)
target_link_libraries(flash_attn_v2_mma_swizzle PUBLIC attention)

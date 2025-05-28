set (sample_attention_baseline_files
    attention_baseline.cu
)

set (sample_minmal_flash_attention_v1_files
    minmal_flash_attention.cu
)

set (sample_flash_attention_v1_files
    flash_attention_v1.cu
)

set (sample_flash_attention_v2_files
    flash_attention_v2.cu
)

set (sample_flash_attention_v3_files
    flash_attention_v3.cu
)
set (sample_flash_attention_v4_files
    flash_attention_v4.cu
)
set (sample_flash_attention_v5_files
    flash_attention_v5.cu
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

add_executable(flash_attention_v1 ${sample_flash_attention_v1_files})
target_compile_features(flash_attention_v1 PUBLIC cxx_std_14)
target_link_libraries(flash_attention_v1 PUBLIC attention)

add_executable(flash_attention_v2 ${sample_flash_attention_v2_files})
target_compile_features(flash_attention_v2 PUBLIC cxx_std_14)
target_link_libraries(flash_attention_v2 PUBLIC attention)

add_executable(flash_attention_v3 ${sample_flash_attention_v3_files})
target_compile_features(flash_attention_v3 PUBLIC cxx_std_14)
target_link_libraries(flash_attention_v3 PUBLIC attention)

add_executable(flash_attention_v4 ${sample_flash_attention_v4_files})
target_compile_features(flash_attention_v4 PUBLIC cxx_std_14)
target_link_libraries(flash_attention_v4 PUBLIC attention)

add_executable(flash_attention_v5 ${sample_flash_attention_v5_files})
target_compile_features(flash_attention_v5 PUBLIC cxx_std_14)
target_link_libraries(flash_attention_v5 PUBLIC attention)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(flash_attention_v5 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
else()
    target_compile_options(flash_attention_v5 PRIVATE -lineinfo)
endif()

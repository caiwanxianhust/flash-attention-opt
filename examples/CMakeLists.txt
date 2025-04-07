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
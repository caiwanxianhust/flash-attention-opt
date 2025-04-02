cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(FLASH_ATTENTION LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDA REQUIRED)


message(STATUS "CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}")  # 输出检测路径‌:ml-citation{ref="1,5" data="citationList"}


set(COMMON_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

set(COMMON_LIB_DIRS
    /usr/local/cuda/lib64
)

include_directories(
  ${COMMON_HEADER_DIRS}
)

message("-- Assign include directories (include_directories=${COMMON_HEADER_DIRS})")
message("-- Assign include directories (common_lib_dirs=${COMMON_LIB_DIRS})")

# add_definitions(-DNDEBUG)

add_subdirectory(flash_attention)
add_subdirectory(examples)
cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
set(CMAKE_CUDA_ARCHITECTURES 89)

project(FLASH_ATTENTION LANGUAGES CXX CUDA)

# 可选：二次锁定防止覆盖
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "" FORCE)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")


# 检查 CUDA 版本
if(CUDART_VERSION LESS 12000)
    message(FATAL_ERROR "CUDA 12.0+ required for sm_89")
endif()

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
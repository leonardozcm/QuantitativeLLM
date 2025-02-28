cmake_minimum_required(VERSION 3.16)
set(OP_NAME name)
project(${OP_NAME} LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories((${PROJECT_SOURCE_DIR}/../../include))

#GPU Benchmark
add_executable(${OP_NAME} ${OP_NAME}.cu)
target_compile_definitions(${OP_NAME} PUBLIC CUDA_EXECUTE)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

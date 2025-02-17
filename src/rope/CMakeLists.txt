cmake_minimum_required(VERSION 3.16)
project(ROPE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories((${PROJECT_SOURCE_DIR}/../../include))

#GPU Benchmark
add_executable(rope rope.cu)
target_compile_definitions(rope PUBLIC CUDA_EXECUTE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

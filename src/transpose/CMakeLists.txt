cmake_minimum_required(VERSION 3.16)
project(TRANSPOSE LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories((${PROJECT_SOURCE_DIR}/../../include))


# CPU Benchmark
add_executable(transpose_cpu transpose_cpu.cpp)

#GPU Benchmark
add_executable(transpose_gpu transpose_gpu.cu)
target_compile_definitions(transpose_gpu PUBLIC CUDA_EXECUTE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

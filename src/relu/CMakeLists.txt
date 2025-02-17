cmake_minimum_required(VERSION 3.16)

set(OPNAME name)

project(${OPNAME} LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 11)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories((${PROJECT_SOURCE_DIR}/../../include))

#GPU Benchmark
add_executable(${OPNAME} ${OPNAME}.cu)
target_compile_definitions(${OPNAME} PUBLIC CUDA_EXECUTE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

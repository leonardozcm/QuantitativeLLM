cmake_minimum_required(VERSION 3.16)
set(OP_NAME sgemm)
project(${OP_NAME} LANGUAGES C CUDA CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include_directories((${PROJECT_SOURCE_DIR}/../../include))

# Global CUDA CXX flags/options
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)


#GPU Benchmark
add_executable(${OP_NAME} ${OP_NAME}.cu)
set_property(TARGET ${OP_NAME} PROPERTY CUDA_ARCHITECTURES OFF)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

# Libraries
target_compile_definitions(${OP_NAME} PUBLIC CUDA_EXECUTE)
target_link_libraries(${OP_NAME}
    PUBLIC
        cublas
)

cmake_minimum_required(VERSION 3.16)
project(TENSOR_BASIC)

set(CMAKE_CXX_STANDARD 11)
include_directories((${PROJECT_SOURCE_DIR}/../../include))

add_executable(tensor_basic tensor_basic.cpp)

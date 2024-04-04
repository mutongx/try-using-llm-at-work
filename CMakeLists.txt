cmake_minimum_required(VERSION 3.20)

project(try_using_llm_at_work)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

set(LLAMA_CUDA ON)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTING OFF)
set(BUILD_TEST OFF)

add_subdirectory(thirdparty/capnproto)
add_subdirectory(thirdparty/fmt)
add_subdirectory(thirdparty/llama.cpp)
add_subdirectory(thirdparty/oniguruma)
add_subdirectory(thirdparty/catch2)

add_subdirectory(proto)
add_subdirectory(backend)

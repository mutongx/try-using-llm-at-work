cmake_minimum_required(VERSION 3.16)
project(try_using_llm_at_work)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(LLAMA_CUBLAS ON)

add_subdirectory(thirdparty/capnproto)
add_subdirectory(thirdparty/fmt)
add_subdirectory(thirdparty/llama.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main capnp fmt llama)

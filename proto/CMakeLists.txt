cmake_minimum_required(VERSION 3.20)

capnp_generate_cpp(PROTO_SRCS PROTO_HDRS common.capnp config.capnp service.capnp)

add_library(proto ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(proto PUBLIC capnp capnp-json capnp-rpc)
target_include_directories(proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

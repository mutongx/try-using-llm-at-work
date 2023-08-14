cmake_minimum_required(VERSION 3.20)

capnp_generate_cpp(PROTO_CONFIG_SRCS PROTO_CONFIG_HDRS proto/config.capnp)
capnp_generate_cpp(PROTO_SERVICE_SRCS PROTO_SERVICE_HDRS proto/service.capnp)

set(CAPNP_WRAPPER_HDRS
        proto/CapnpMessage.h
        proto/CapnpUtilities.h
)

set(LLAMA_WRAPPER_SRCS
        llama/LlamaScope.cpp
        llama/LlamaParams.cpp
        llama/LlamaModel.cpp
        llama/LlamaContext.cpp
        llama/LlamaTokenizer.cpp)

set(LLAMA_WRAPPER_HDRS
        llama/LlamaScope.h
        llama/LlamaParams.h
        llama/LlamaModel.h
        llama/LlamaContext.h
        llama/LlamaTokenizer.h)

add_executable(play
        ${PROTO_CONFIG_SRCS} ${PROTO_CONFIG_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CAPNP_WRAPPER_HDRS}
        Config.h Config.cpp
        play.cpp
)

target_link_libraries(play capnp capnp-json fmt llama)
target_include_directories(play PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

add_executable(service
        ${PROTO_CONFIG_SRCS} ${PROTO_CONFIG_HDRS}
        ${PROTO_SERVICE_SRCS} ${PROTO_SERVICE_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CAPNP_WRAPPER_HDRS}
        Config.h Config.cpp
        service.cpp
)

target_link_libraries(service capnp capnp-json capnp-rpc fmt llama)
target_include_directories(service PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

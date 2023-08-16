cmake_minimum_required(VERSION 3.20)

capnp_generate_cpp(PROTO_CONFIG_SRCS PROTO_CONFIG_HDRS proto/config.capnp)
capnp_generate_cpp(PROTO_SERVICE_SRCS PROTO_SERVICE_HDRS proto/service.capnp)

set(CAPNP_WRAPPER_HDRS
        proto/CapnpMessage.h
        proto/CapnpUtilities.h)

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

set(CONFIG_SRCS
        config/Config.cpp)

set(CONFIG_HDRS
        config/Config.h)

set(SERVICE_SRCS
        service/AppServer.cpp
        service/ModelServer.cpp
        service/TokenizerServer.cpp)

set(SERVICE_HDRS
        service/AppServer.h
        service/ModelServer.h
        service/TokenizerServer.h)

add_executable(play
        ${CAPNP_WRAPPER_HDRS}
        ${PROTO_CONFIG_SRCS} ${PROTO_CONFIG_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        play.cpp
)

target_link_libraries(play capnp capnp-json fmt llama)
target_include_directories(play PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_executable(service
        ${CAPNP_WRAPPER_HDRS}
        ${PROTO_CONFIG_SRCS} ${PROTO_CONFIG_HDRS}
        ${PROTO_SERVICE_SRCS} ${PROTO_SERVICE_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        ${SERVICE_HDRS} ${SERVICE_SRCS}
        service.cpp
)

target_link_libraries(service capnp capnp-json capnp-rpc fmt llama)
target_include_directories(service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

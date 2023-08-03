cmake_minimum_required(VERSION 3.20)

capnp_generate_cpp(PROTO_CONFIG_SRCS PROTO_CONFIG_HDRS proto/config.capnp)

add_executable(play
        ${PROTO_CONFIG_SRCS}
        ${PROTO_CONFIG_HDRS}
        proto/CapnpMessage.h
        llama/LlamaScope.cpp
        llama/LlamaScope.h
        llama/LlamaParams.cpp
        llama/LlamaParams.h
        llama/LlamaModel.cpp
        llama/LlamaModel.h
        llama/LlamaContext.cpp
        llama/LlamaContext.h
        llama/LlamaTokenizer.cpp
        llama/LlamaTokenizer.h
        Config.h Config.cpp
        play.cpp

)
target_link_libraries(play capnp capnp-json fmt llama)
target_include_directories(play PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

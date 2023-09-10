cmake_minimum_required(VERSION 3.20)

configure_file(../thirdparty/llama.cpp/ggml-metal.metal ggml-metal.metal COPYONLY)

set(CAPNP_WRAPPER_HDRS
        capnp/CapnpMessage.h
        capnp/CapnpUtilities.h)

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
        service/ContextServer.cpp
        service/TokenizerServer.cpp)

set(SERVICE_HDRS
        service/AppServer.h
        service/ModelServer.h
        service/ContextServer.h
        service/TokenizerServer.h)

add_executable(backend
        ${CAPNP_WRAPPER_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        ${SERVICE_HDRS} ${SERVICE_SRCS}
        main.cpp)

target_link_libraries(backend proto fmt llama)
target_include_directories(backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_SRCS
        llama/LlamaModel_test.cpp
        llama/LlamaTokenizer_test.cpp)

add_executable(backend_test
        ${CAPNP_WRAPPER_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        ${TEST_SRCS}
)

target_link_libraries(backend_test PRIVATE Catch2::Catch2WithMain proto llama)
target_include_directories(backend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

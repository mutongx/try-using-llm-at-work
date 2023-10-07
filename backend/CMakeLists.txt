cmake_minimum_required(VERSION 3.20)

configure_file(../thirdparty/llama.cpp/ggml-metal.metal ggml-metal.metal COPYONLY)

set(INITIALIZE_SRCS
        initialize.cpp
)

set(INITIALIZE_HDRS
        initialize.h
)

set(CAPNP_WRAPPER_HDRS
        capnp/CapnpMessage.h
        capnp/CapnpUtilities.h
)

set(LLAMA_WRAPPER_SRCS
        llama/LlamaParams.cpp
        llama/LlamaModel.cpp
        llama/LlamaContext.cpp
        llama/LlamaTokenizer.cpp
)

set(LLAMA_WRAPPER_HDRS
        llama/LlamaParams.h
        llama/LlamaModel.h
        llama/LlamaContext.h
        llama/LlamaTokenizer.h
)

set(CONFIG_SRCS
        config/Config.cpp
)

set(CONFIG_HDRS
        config/Config.h
)

set(SERVICE_SRCS
        service/AppServer.cpp
        service/ModelServer.cpp
        service/ContextServer.cpp
        service/TokenizerServer.cpp
)

set(SERVICE_HDRS
        service/AppServer.h
        service/ModelServer.h
        service/ContextServer.h
        service/TokenizerServer.h
)

add_executable(backend
        ${INITIALIZE_SRCS} ${INITIALIZE_HDRS}
        ${CAPNP_WRAPPER_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        ${SERVICE_SRCS} ${SERVICE_HDRS}
        main.cpp
)

target_link_libraries(backend proto fmt llama onig)
target_include_directories(backend PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set(TEST_SRCS
        llama/LlamaModel_test.cpp
        llama/LlamaTokenizer_test.cpp
        llama/LlamaContext_test.cpp
)

add_executable(backend_test
        ${INITIALIZE_SRCS} ${INITIALIZE_HDRS}
        ${CAPNP_WRAPPER_HDRS}
        ${LLAMA_WRAPPER_SRCS} ${LLAMA_WRAPPER_HDRS}
        ${CONFIG_SRCS} ${CONFIG_HDRS}
        ${TEST_SRCS}
        test_main.cpp
)

target_link_libraries(backend_test PRIVATE Catch2::Catch2 proto fmt llama onig)
target_include_directories(backend_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

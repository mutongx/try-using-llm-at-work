#ifndef MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H

#include "llama/LlamaContext.h"
#include "llama/LlamaParams.h"

#include "proto.capnp.h"

namespace muton::playground::llm {

class ContextServer : public proto::Context::Server {
 public:
  ContextServer(LlamaParams& params, LlamaModel& model);
  ContextServer(ContextServer&&) = delete;
  ContextServer(ContextServer const&) = delete;
  ContextServer& operator=(ContextServer&&) = delete;
  ContextServer& operator=(ContextServer const&) = delete;
  ~ContextServer() = default;

 private:
  LlamaContext context_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H

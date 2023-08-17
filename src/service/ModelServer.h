#ifndef MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"
#include "llama/LlamaTokenizer.h"

#include "proto/service.capnp.h"

namespace muton::playground::llm {

class ModelServer : public proto::Model::Server {
 public:
  ModelServer(LlamaModel& model_);
  ModelServer(ModelServer&&) = delete;
  ModelServer(ModelServer const&) = delete;
  ModelServer& operator=(ModelServer&&) = delete;
  ModelServer& operator=(ModelServer const&) = delete;
  ~ModelServer() = default;

  kj::Promise<void> getTokenizer(GetTokenizerContext context);

 private:
  LlamaModel& model_;
  LlamaTokenizer tokenizer_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H
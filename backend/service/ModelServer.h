#ifndef MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"

#include "service.capnp.h"

namespace muton::playground::llm {

class ModelServer : public proto::Model::Server {
 public:
  ModelServer(LlamaParams& params, LlamaModel& model);
  ModelServer(ModelServer&&) = delete;
  ModelServer(ModelServer const&) = delete;
  ModelServer& operator=(ModelServer&&) = delete;
  ModelServer& operator=(ModelServer const&) = delete;
  ~ModelServer() = default;

  kj::Promise<void> newTokenizer(NewTokenizerContext context) override;
  kj::Promise<void> newContext(NewContextContext context) override;

 private:
  LlamaParams& params_;
  LlamaModel& model_;
  LlamaVocabulary vocabulary_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_MODEL_SERVER_H

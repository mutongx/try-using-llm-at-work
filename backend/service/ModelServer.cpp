#include "ModelServer.h"

#include "ContextServer.h"
#include "TokenizerServer.h"

namespace muton::playground::llm {

ModelServer::ModelServer(LlamaParams& params, LlamaModel& model)
    : params_(params), model_(model), vocabulary_(model.GetVocabulary()) {}

kj::Promise<void> ModelServer::newTokenizer(NewTokenizerContext context) {
  context.getResults().setTokenizer(kj::heap<TokenizerServer>(model_));
  return kj::READY_NOW;
}

kj::Promise<void> ModelServer::newContext(NewContextContext context) {
  context.getResults().setContext(kj::heap<ContextServer>(params_, model_, vocabulary_));
  return kj::READY_NOW;
}

}  // namespace muton::playground::llm

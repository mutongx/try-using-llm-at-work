#include "ModelServer.h"
#include "TokenizerServer.h"

#include <fmt/format.h>

namespace muton::playground::llm {

ModelServer::ModelServer(LlamaModel& model) : model_(model), tokenizer_(model) {}

kj::Promise<void> ModelServer::getTokenizer(GetTokenizerContext context) {
  context.getResults().setTokenizer(kj::heap<TokenizerServer>(tokenizer_));
  return kj::READY_NOW;
}

}  // namespace muton::playground::llm

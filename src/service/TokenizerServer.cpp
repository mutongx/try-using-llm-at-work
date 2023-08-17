#include "TokenizerServer.h"

#include "proto/CapnpUtilities.h"

namespace muton::playground::llm {

TokenizerServer::TokenizerServer(LlamaModel& model) : tokenizer_(model) {}

kj::Promise<void> TokenizerServer::tokenize(TokenizeContext context) {
  auto text_proto = context.getParams().getText();
  auto text_view = std::string_view(text_proto.cStr(), text_proto.size());
  auto tokenize_result = tokenizer_.Tokenize(text_view);
  auto response = context.getResults().getTokens();
  response.setSize(tokenize_result.size);
  response.adoptTokenId(CreateCapnpList<int32_t>(context.getResultsOrphanage(), tokenize_result.token_id));
  response.adoptTokenPos(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), tokenize_result.token_pos));
  response.adoptTokenSize(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), tokenize_result.token_size));
  return kj::READY_NOW;
};

}  // namespace muton::playground::llm

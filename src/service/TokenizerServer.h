#ifndef MUTON_PLAYGROUND_LLM_SERVICE_TOKENIZER_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_TOKENIZER_SERVER_H

#include "llama/LlamaTokenizer.h"

#include "proto/service.capnp.h"

namespace muton::playground::llm {

class TokenizerServer : public proto::Tokenizer::Server {
 public:
  TokenizerServer(LlamaTokenizer& tokenizer);
  TokenizerServer(TokenizerServer&&) = delete;
  TokenizerServer(TokenizerServer const&) = delete;
  TokenizerServer& operator=(TokenizerServer&&) = delete;
  TokenizerServer& operator=(TokenizerServer const&) = delete;
  ~TokenizerServer() = default;

  kj::Promise<void> tokenize(TokenizeContext context);

 private:
  LlamaTokenizer& tokenizer_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_TOKENIZER_SERVER_H

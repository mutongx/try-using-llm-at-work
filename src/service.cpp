#include <capnp/ez-rpc.h>

#include "Config.h"

#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"
#include "llama/LlamaTokenizer.h"

#include "proto/CapnpUtilities.h"
#include "proto/service.capnp.h"

namespace muton::playground::llm {

class TokenizerImpl final : public proto::Tokenizer::Server {
 public:
  TokenizerImpl(LlamaTokenizer& tokenizer) : tokenizer_(tokenizer) {}
  kj::Promise<void> tokenize(TokenizeContext context) override {
    auto text_proto = context.getParams().getText();
    auto text_view = std::string_view(text_proto.cStr(), text_proto.size());
    auto tokenize_result = tokenizer_.Tokenize(text_view);
    auto response = context.getResults().getTokens();
    response.setSize(tokenize_result.size);
    response.adoptTokenId(CreateCapnpList<int32_t>(context.getResultsOrphanage(), tokenize_result.token_id));
    response.adoptTokenPos(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), tokenize_result.token_pos));
    response.adoptTokenSize(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), tokenize_result.token_size));
    return kj::READY_NOW;
  }

 private:
  LlamaTokenizer tokenizer_;
};

}  // namespace muton::playground::llm

int main() {
  auto config = muton::playground::llm::Config::Read();
  auto cfg_model = config->getModel();
  auto cfg_params = config->getParams();
  auto cfg_eval = config->getEval();
  auto cfg_predict = config->getPredict();

  muton::playground::llm::LlamaScope backend(false);
  muton::playground::llm::LlamaParams params{cfg_params};
  muton::playground::llm::LlamaModel model{cfg_model.cStr(), params};
  muton::playground::llm::LlamaTokenizer tokenizer{model};

  capnp::EzRpcServer server(kj::heap<muton::playground::llm::TokenizerImpl>(tokenizer), "127.0.0.1:2333");
  auto& waitScope = server.getWaitScope();
  kj::NEVER_DONE.wait(waitScope);
}

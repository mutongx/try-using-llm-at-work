#include "TokenizerServer.h"

#include "capnp/CapnpUtilities.h"

namespace muton::playground::llm {

TokenizerServer::TokenizerServer(LlamaModel& model) : tokenizer_(model) {}

kj::Promise<void> TokenizerServer::tokenize(TokenizeContext context) {
  class TokenizeResultServer : public proto::Tokens::Server {
   public:
    TokenizeResultServer(LlamaTokenizer::TokenizeResult result) : result_(std::move(result)) {}
    kj::Promise<void> getSize(GetSizeContext context) override {
      context.getResults().setResult(result_.size);
      return kj::READY_NOW;
    }
    kj::Promise<void> getTokenId(GetTokenIdContext context) override {
      context.getResults().adoptResult(CreateCapnpList<int32_t>(context.getResultsOrphanage(), result_.token_id));
      return kj::READY_NOW;
    }
    kj::Promise<void> getTokenPos(GetTokenPosContext context) override {
      context.getResults().adoptResult(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), result_.token_pos));
      return kj::READY_NOW;
    }
    kj::Promise<void> getTokenSize(GetTokenSizeContext context) override {
      context.getResults().adoptResult(CreateCapnpList<uint32_t>(context.getResultsOrphanage(), result_.token_size));
      return kj::READY_NOW;
    }
    kj::Promise<void> getInternalPtr(GetInternalPtrContext context) override {
      context.getResults().setPtr(reinterpret_cast<uint64_t>(&result_));
      return kj::READY_NOW;
    }

   private:
    LlamaTokenizer::TokenizeResult result_;
  };

  auto text = context.getParams().getText();
  auto text_view = std::string_view(text.cStr(), text.size());
  context.getResults().setTokens(kj::heap<TokenizeResultServer>(tokenizer_.Tokenize(text_view)));
  return kj::READY_NOW;
};

}  // namespace muton::playground::llm

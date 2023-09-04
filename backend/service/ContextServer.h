#ifndef MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H

#include "llama/LlamaContext.h"
#include "llama/LlamaParams.h"

#include "service.capnp.h"

namespace muton::playground::llm {

class ContextServer : public proto::Context::Server {
 public:
  ContextServer(LlamaParams& params, LlamaModel& model);
  ContextServer(ContextServer&&) = delete;
  ContextServer(ContextServer const&) = delete;
  ContextServer& operator=(ContextServer&&) = delete;
  ContextServer& operator=(ContextServer const&) = delete;
  ~ContextServer() = default;

  kj::Promise<void> nop(NopContext context) override;
  kj::Promise<void> feedTokens(FeedTokensContext context) override;
  kj::Promise<void> feedToken(FeedTokenContext context) override;
  kj::Promise<void> feedBos(FeedBosContext context) override;
  kj::Promise<void> feedEos(FeedEosContext context) override;
  kj::Promise<void> eval(EvalContext context) override;
  kj::Promise<void> predict(PredictContext context) override;
  kj::Promise<void> predictUntilEos(PredictUntilEosContext context) override;

 private:
  kj::Promise<bool> feedTokensInternal(proto::Tokens::Client tokens, int32_t begin, int32_t end);
  kj::Promise<void> predictUntilEosInternal(proto::Context::PredictCallback::Client callback,
                                            proto::EvalOption::Reader eval_option,
                                            proto::PredictOption::Reader predict_option);

  using PredictRequest =
      capnp::Request<proto::Context::PredictCallback::CallbackParams, proto::Context::PredictCallback::CallbackResults>;
  PredictRequest newPredictRequest(proto::Context::PredictCallback::Client& callback, llama_token token);

  LlamaModel& model_;
  LlamaContext context_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_CONTEXT_SERVER_H

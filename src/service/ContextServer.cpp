#include "ContextServer.h"

#include "llama/LlamaTokenizer.h"

namespace muton::playground::llm {

ContextServer::ContextServer(LlamaParams& params, LlamaModel& model) : model_(model), context_(params, model) {}

kj::Promise<void> ContextServer::nop(proto::Context::Server::NopContext context) {
  auto results = context.getResults();
  results.setSuccess(true);
  results.setContext(this->thisCap());
  return kj::READY_NOW;
}

kj::Promise<void> ContextServer::feedTokens(proto::Context::Server::FeedTokensContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return feedTokensInternal(params.getTokens(), params.getBegin(), params.getEnd())
        .then([this, results](bool success) mutable {
          results.setSuccess(success);
          results.setContext(this->thisCap());
        });
  });
}

kj::Promise<void> ContextServer::feedToken(proto::Context::Server::FeedTokenContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.Feed(params.getToken());
    results.setSuccess(success);
    results.setContext(this->thisCap());
  });
}

kj::Promise<void> ContextServer::feedBos(proto::Context::Server::FeedBosContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.FeedBos();
    results.setSuccess(success);
    results.setContext(this->thisCap());
  });
}

kj::Promise<void> ContextServer::feedEos(proto::Context::Server::FeedEosContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.FeedEos();
    results.setSuccess(success);
    results.setContext(this->thisCap());
  });
}

kj::Promise<void> ContextServer::eval(proto::Context::Server::EvalContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.Eval(params.getOption());
    results.setSuccess(success);
    results.setContext(this->thisCap());
  });
}

kj::Promise<void> ContextServer::predict(proto::Context::Server::PredictContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return predictInternal(params.getCallback(), params.getOption()).then([this, results](bool success) mutable {
      results.setSuccess(success);
      results.setContext(this->thisCap());
    });
  });
}

kj::Promise<void> ContextServer::predictUntilEos(proto::Context::Server::PredictUntilEosContext context) {
  return kj::READY_NOW;
}

kj::Promise<bool> ContextServer::feedTokensInternal(proto::Tokens::Client tokens, int32_t begin, int32_t end) {
  return tokens.getInternalPtrRequest().send().then([this, begin, end](auto ptr) {
    auto& token_id = reinterpret_cast<LlamaTokenizer::TokenizeResult*>(ptr.getPtr())->token_id;
    auto token_size = static_cast<std::ptrdiff_t>(token_id.size());
    auto begin_it = token_id.begin();
    auto end_it = token_id.begin();
    if (begin > token_id.size()) {
      begin_it += token_size;
    } else if (begin >= 0) {
      begin_it += begin;
    }
    if (end < 0 || end > token_id.size()) {
      end_it += token_size;
    } else {
      end_it += end;
    }
    if (begin_it > end_it) {
      return false;
    }
    return context_.Feed({begin_it, end_it});
  });
}

kj::Promise<bool> ContextServer::predictInternal(proto::Context::PredictCallback::Client callback,
                                                 proto::PredictOption::Reader option) {
  auto token = context_.Predict(option);
  auto request = callback.callbackRequest();
  auto result = request.getToken();
  result.setId(token);
  if (token == model_.GetBos()) {
    result.setType(proto::Token::TokenType::BEGIN_OF_STREAM);
  } else if (token == model_.GetEos()) {
    result.setType(proto::Token::TokenType::END_OF_STREAM);
  } else {
    result.setType(proto::Token::TokenType::NORMAL);
  }
  result.setStr(model_.GetTokenString(token));
  return request.send().then([](auto) {
    return true;
  });
}

}  // namespace muton::playground::llm

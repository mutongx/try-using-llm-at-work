#include "ContextServer.h"

#include "llama/LlamaTokenizer.h"

namespace muton::playground::llm {

ContextServer::ContextServer(LlamaParams& params, LlamaModel& model) : model_(model), context_(params, model) {}

kj::Promise<void> ContextServer::nop(proto::Context::Server::NopContext context) {
  auto results = context.getResults();
  results.setSuccess(true);
  results.setContext(thisCap());
  return kj::READY_NOW;
}

kj::Promise<void> ContextServer::feedTokens(proto::Context::Server::FeedTokensContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return feedTokensInternal(params.getTokens(), params.getBegin(), params.getEnd())
        .then([this, results](bool success) mutable {
          results.setSuccess(success);
          results.setContext(thisCap());
        });
  });
}

kj::Promise<void> ContextServer::feedToken(proto::Context::Server::FeedTokenContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.Feed(params.getToken());
    results.setSuccess(success);
    results.setContext(thisCap());
  });
}

kj::Promise<void> ContextServer::feedBos(proto::Context::Server::FeedBosContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.FeedBos();
    results.setSuccess(success);
    results.setContext(thisCap());
  });
}

kj::Promise<void> ContextServer::feedEos(proto::Context::Server::FeedEosContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    auto success = context_.FeedEos();
    results.setSuccess(success);
    results.setContext(thisCap());
  });
}

kj::Promise<void> ContextServer::eval(proto::Context::Server::EvalContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return evalInternal(params.getOption()).then([this, results](bool success) mutable {
      results.setSuccess(success);
      results.setContext(thisCap());
    });
  });
}

kj::Promise<void> ContextServer::predict(proto::Context::Server::PredictContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return predictInternal(params.getCallback(), params.getOption()).then([this, results]() mutable {
      results.setSuccess(true);
      results.setContext(thisCap());
    });
  });
}

kj::Promise<void> ContextServer::predictUntilEos(proto::Context::Server::PredictUntilEosContext context) {
  return kj::READY_NOW.operator kj::Promise<void>().then([this, context]() mutable {
    auto params = context.getParams();
    auto results = context.getResults();
    return predictUntilEosInternal(params.getCallback(), params.getEvalOption(), params.getPredictOption())
        .then([this, results]() mutable {
          results.setSuccess(true);
          results.setContext(thisCap());
        });
  });
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

kj::Promise<bool> ContextServer::evalInternal(proto::EvalOption::Reader eval_option) {
  auto left = context_.Eval(eval_option);
  if (left > 0) {
    return kj::READY_NOW.operator kj::Promise<void>().then([this, eval_option = kj::mv(eval_option)]() {
      return evalInternal(kj::mv(eval_option));
    });
  }
  return left == 0;
}

kj::Promise<void> ContextServer::predictInternal(proto::Context::PredictCallback::Client callback,
                                                 proto::PredictOption::Reader predict_option) {
  // TODO: Add mirostat predict when ready
  auto token = context_.Predict(predict_option);
  return newPredictCallbackRequest(callback, token).send().ignoreResult();
}

kj::Promise<void> ContextServer::predictUntilEosInternal(proto::Context::PredictCallback::Client callback,
                                                         proto::EvalOption::Reader eval_option,
                                                         proto::PredictOption::Reader predict_option) {
  kj::Promise<void> next_request{kj::READY_NOW};
  bool next_iter{false};
  // Runs eval.
  auto left = context_.Eval(eval_option);
  if (left > 0) {  // Eval() is not completed.
    next_iter = true;
  } else {
    // Runs predict.
    auto token = context_.Predict(predict_option);
    if (!context_.Feed(token) || (token == model_.GetEos())) {
      // If context is full or token is EOS, stop generating.
      next_request = callback.doneRequest().send().ignoreResult();
    } else {
      // Should continue generating.
      next_request = newPredictCallbackRequest(callback, token).send().ignoreResult();
      next_iter = true;
    }
  }
  if (next_iter) {
    return next_request.then([this,
                              callback = kj::mv(callback),
                              eval_option = kj::mv(eval_option),
                              predict_option = kj::mv(predict_option)]() mutable {
      return predictUntilEosInternal(kj::mv(callback), kj::mv(eval_option), kj::mv(predict_option));
    });
  }
  return next_request;
}

ContextServer::PredictCallbackRequest ContextServer::newPredictCallbackRequest(PredictCallback::Client callback,
                                                                               llama_token token) {
  auto request = callback.callbackRequest();
  auto result = request.getToken();
  result.setId(token);
  result.setStr(model_.GetTokenPiece(token));
  return request;
}

}  // namespace muton::playground::llm

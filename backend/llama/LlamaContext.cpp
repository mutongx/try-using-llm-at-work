#include "LlamaContext.h"

#include <cstring>

namespace muton::playground::llm {

LlamaContext::LlamaContext(LlamaParams const& params, LlamaModel const& model)
    : context_size_(params.GetContextParams().n_ctx),
      tokens_(context_size_, 0),
      context_(llama_new_context_with_model(model.Get(), params.GetContextParams())) {}

LlamaContext::LlamaContext(LlamaContext&& another) noexcept {
  MoveFrom(std::move(another));
}

LlamaContext& LlamaContext::operator=(LlamaContext&& another) noexcept {
  MoveFrom(std::move(another));
  return *this;
}

void LlamaContext::MoveFrom(LlamaContext&& another) noexcept {
  context_size_ = another.context_size_;
  tokens_ = std::move(another.tokens_);
  tokens_begin_ = another.tokens_begin_;
  tokens_size_ = another.tokens_size_;
  tokens_eval_ = another.tokens_eval_;
  context_ = another.context_;
  another.context_ = nullptr;
}

bool LlamaContext::FeedBos() {
  auto bos = llama_token_bos(context_);
  return Feed(std::span<llama_token>(&bos, 1));
}

bool LlamaContext::FeedEos() {
  auto eos = llama_token_eos(context_);
  return Feed(std::span<llama_token>(&eos, 1));
}

bool LlamaContext::Feed(llama_token token_pending) {
  return Feed(std::span<llama_token>(&token_pending, 1));
}

bool LlamaContext::Feed(std::span<llama_token> tokens_pending) {
  if (tokens_size_ + tokens_pending.size() > context_size_) {
    return false;
  }
  memcpy(tokens_.data() + tokens_size_, tokens_pending.data(), tokens_pending.size() * sizeof(llama_token));
  tokens_size_ += tokens_pending.size();
  return true;
}

bool LlamaContext::Eval(proto::EvalOption::Reader option) {
  size_t to_eval = tokens_size_ - tokens_eval_;
  while (to_eval > 0) {
    size_t current_batch_size = std::min(to_eval, static_cast<size_t>(option.getBatchSize()));
    if (llama_eval(context_,
                   tokens_.data() + tokens_eval_,
                   static_cast<int>(current_batch_size),
                   static_cast<int>(tokens_eval_),
                   static_cast<int>(option.getThreadCount())) != 0) {
      return false;
    }
    tokens_eval_ += current_batch_size;
    to_eval -= current_batch_size;
  }
  return true;
}

llama_token LlamaContext::Predict(proto::PredictOption::Reader option) {
  auto* logits = llama_get_logits(context_);
  auto vocab_size = llama_n_vocab(context_);
  std::vector<llama_token_data> candidates(vocab_size);
  for (llama_token token_id = 0; token_id < vocab_size; ++token_id) {
    candidates[token_id].id = token_id;
    candidates[token_id].logit = logits[token_id];
    candidates[token_id].p = 0.0F;
  }
  llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

  size_t repeat_penalty_size =
      std::min(std::min(tokens_size_, static_cast<size_t>(option.getRepeatPenaltySize())), context_size_);
  llama_sample_repetition_penalty(context_,
                                  &candidates_p,
                                  tokens_.data() + tokens_size_ - repeat_penalty_size,
                                  repeat_penalty_size,
                                  option.getRepeatPenalty());
  llama_sample_frequency_and_presence_penalties(context_,
                                                &candidates_p,
                                                tokens_.data() + tokens_size_ - repeat_penalty_size,
                                                repeat_penalty_size,
                                                option.getAlphaFrequency(),
                                                option.getAlphaPresence());

  if (option.getTemperature() <= 0) {
    return llama_sample_token_greedy(context_, &candidates_p);
  }

  llama_sample_top_k(context_, &candidates_p, static_cast<int>(option.getTopK()), 1);
  llama_sample_tail_free(context_, &candidates_p, option.getTailFreeZ(), 1);
  llama_sample_typical(context_, &candidates_p, option.getTypicalP(), 1);
  llama_sample_top_p(context_, &candidates_p, option.getTopP(), 1);
  llama_sample_temperature(context_, &candidates_p, option.getTemperature());
  return llama_sample_token(context_, &candidates_p);
}

LlamaContext::~LlamaContext() {
  if (context_ != nullptr) {
    llama_free(context_);
  }
}

}  // namespace muton::playground::llm
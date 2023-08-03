#include "LlamaContext.h"

#include <cstring>

namespace muton::playground::llm {

LlamaContext::LlamaContext(LlamaModel const& model, LlamaParams const& params)
    : context_(llama_new_context_with_model(model, params)), context_size_(params->n_ctx), tokens_(context_size_, 0) {}

LlamaContext::LlamaContext(LlamaContext&& another) noexcept {
  MoveFrom(std::move(another));
}

LlamaContext& LlamaContext::operator=(LlamaContext&& another) noexcept {
  MoveFrom(std::move(another));
  return *this;
}

void LlamaContext::MoveFrom(LlamaContext&& another) noexcept {
  context_ = another.context_;
  context_size_ = another.context_size_;
  tokens_ = std::move(another.tokens_);
  tokens_begin_ = another.tokens_begin_;
  tokens_size_ = another.tokens_size_;
  tokens_eval_ = another.tokens_eval_;
  another.context_ = nullptr;
}

LlamaContext::operator llama_context*() const {
  return context_;
}

bool LlamaContext::FeedBos() {
  auto bos = llama_token_bos();
  return Feed(std::span<llama_token>(&bos, 1));
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

bool LlamaContext::Eval(LlamaContext::EvalOption option) {
  size_t to_eval = tokens_size_ - tokens_eval_;
  while (to_eval > 0) {
    size_t current_batch_size = std::min(to_eval, option.batch_size);
    if (llama_eval(context_,
                   tokens_.data() + tokens_eval_,
                   static_cast<int>(current_batch_size),
                   static_cast<int>(tokens_eval_),
                   static_cast<int>(option.thread_count)) != 0) {
      return false;
    }
    tokens_eval_ += current_batch_size;
    to_eval -= current_batch_size;
  }
  return true;
}

llama_token LlamaContext::Predict(LlamaContext::PredictOption option) {
  auto* logits = llama_get_logits(context_);
  auto vocab_size = llama_n_vocab(context_);
  std::vector<llama_token_data> candidates(vocab_size);
  for (llama_token token_id = 0; token_id < vocab_size; ++token_id) {
    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0F});
  }
  llama_token_data_array candidates_p = {candidates.data(), candidates.size(), false};

  size_t repeat_penalty_size = std::min(std::min(tokens_size_, option.repeat_penalty_size), context_size_);
  llama_sample_repetition_penalty(context_,
                                  &candidates_p,
                                  tokens_.data() + tokens_size_ - repeat_penalty_size,
                                  repeat_penalty_size,
                                  option.repeat_penalty);
  llama_sample_frequency_and_presence_penalties(context_,
                                                &candidates_p,
                                                tokens_.data() + tokens_size_ - repeat_penalty_size,
                                                repeat_penalty_size,
                                                option.alpha_frequency,
                                                option.alpha_presence);
  llama_sample_top_k(context_, &candidates_p, option.top_k, 1);
  llama_sample_tail_free(context_, &candidates_p, option.tail_free_z, 1);
  llama_sample_typical(context_, &candidates_p, option.typical_p, 1);
  llama_sample_top_p(context_, &candidates_p, option.top_p, 1);
  llama_sample_temperature(context_, &candidates_p, option.temperature);
  return llama_sample_token(context_, &candidates_p);
}

LlamaContext::~LlamaContext() {
  if (context_ != nullptr) {
    llama_free(context_);
  }
}

}  // namespace muton::playground::llm
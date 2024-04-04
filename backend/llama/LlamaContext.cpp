#include "LlamaContext.h"

#include <cstring>

namespace muton::playground::llm {

LlamaContext::LlamaContext(LlamaParams const& params, LlamaModel const& model)
    : context_size_(params.GetContextParams().n_ctx),
      tokens_(context_size_, 0),
      context_(llama_new_context_with_model(model.Get(), params.GetContextParams())) {}

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

ssize_t LlamaContext::Eval(proto::EvalOption::Reader option) {
  size_t to_eval = std::min(tokens_size_ - tokens_eval_, static_cast<size_t>(option.getBatchSize()));
  if (to_eval > 0) {
    llama_seq_id seq_id_val = 0;
    std::vector<llama_pos> pos(to_eval);
    std::vector<int32_t> n_seq_id(to_eval);
    std::vector<llama_seq_id*> seq_id(to_eval);
    for (size_t i = 0; i < to_eval; ++i) {
      pos[i] = tokens_eval_ + i;
      n_seq_id[i] = 1;
      seq_id[i] = &seq_id_val;
    }
    llama_batch batch{
        .n_tokens = static_cast<int32_t>(to_eval),
        .token = tokens_.data() + tokens_eval_,
        .embd = nullptr,
        .pos = pos.data(),
        .n_seq_id = n_seq_id.data(),
        .seq_id = seq_id.data(),
        .logits = nullptr,
    };
    if (llama_decode(context_, batch) != 0) {
      return -1;
    }
  }
  tokens_eval_ += to_eval;
  return static_cast<ssize_t>(tokens_size_ - tokens_eval_);
}

llama_token LlamaContext::Predict(proto::PredictOption::Reader option) {
  auto* logits = llama_get_logits(context_);
  auto vocab_size = llama_n_vocab(llama_get_model(context_));
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
  llama_sample_temp(context_, &candidates_p, option.getTemperature());
  return llama_sample_token(context_, &candidates_p);
}

LlamaContext::~LlamaContext() {
  if (context_ != nullptr) {
    llama_free(context_);
  }
}

}  // namespace muton::playground::llm
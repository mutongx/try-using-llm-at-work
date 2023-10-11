#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

#include <span>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

#include "LlamaModel.h"
#include "LlamaParams.h"

#include "common.capnp.h"

namespace muton::playground::llm {

class LlamaContext {
 public:
  LlamaContext(LlamaParams const& params, LlamaModel const& model);
  LlamaContext(LlamaContext const&) = delete;
  LlamaContext(LlamaContext&& another) = delete;
  LlamaContext& operator=(LlamaContext const&) = delete;
  LlamaContext& operator=(LlamaContext&& another) = delete;
  ~LlamaContext();

  [[nodiscard]] llama_context* Get() {
    return context_;
  }

  [[nodiscard]] llama_token GetBos() const;
  [[nodiscard]] llama_token GetEos() const;

  [[nodiscard]] bool Feed(llama_token token_pending);
  [[nodiscard]] bool Feed(std::span<llama_token> tokens_pending);
  [[nodiscard]] bool FeedBos();
  [[nodiscard]] bool FeedEos();

  [[nodiscard]] ssize_t Eval(proto::EvalOption::Reader option);
  [[nodiscard]] llama_token Predict(proto::PredictOption::Reader option);

 private:

  size_t context_size_{};

  // For data stored in llama_batch
  std::vector<llama_token> tokens_{};
  std::vector<llama_pos> tokens_pos_{};
  std::vector<llama_seq_id> tokens_seq_id_{};

  // To track current feed / eval progress
  size_t tokens_begin_{};
  size_t tokens_size_{};
  size_t tokens_eval_{};

  llama_context* context_{};
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

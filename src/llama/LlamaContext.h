#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

#include <span>
#include <string>
#include <thread>
#include <vector>

#include <llama.h>

#include "LlamaModel.h"
#include "LlamaParams.h"

namespace muton::playground::llm {

class LlamaContext {
 public:
  LlamaContext(LlamaModel const& model, LlamaParams const& params);
  LlamaContext(LlamaContext const&) = delete;
  LlamaContext(LlamaContext&& another) noexcept;
  LlamaContext& operator=(LlamaContext const&) = delete;
  LlamaContext& operator=(LlamaContext&& another) noexcept;
  ~LlamaContext();

  operator llama_context*() const;

  bool FeedBos();
  bool Feed(llama_token token_pending);
  bool Feed(std::span<llama_token> tokens_pending);

  struct EvalOption {
    size_t batch_size;
    size_t thread_count;
  };

  bool Eval(EvalOption option);

  struct PredictOption {
    size_t repeat_penalty_size;
    float repeat_penalty;
    float alpha_presence;
    float alpha_frequency;
    int top_k;
    float tail_free_z;
    float typical_p;
    float top_p;
    float temperature;
  };
  llama_token Predict(PredictOption option);

 private:
  void MoveFrom(LlamaContext&& another) noexcept;

  llama_context* context_{};
  size_t context_size_{};

  std::vector<llama_token> tokens_{};
  size_t tokens_begin_{};
  size_t tokens_size_{};
  size_t tokens_eval_{};
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

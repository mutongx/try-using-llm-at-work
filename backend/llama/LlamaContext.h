#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

#include <span>
#include <string>
#include <thread>
#include <vector>

#include <llama.h>

#include "LlamaModel.h"
#include "LlamaParams.h"

#include "common.capnp.h"

namespace muton::playground::llm {

class LlamaContext {
 public:
  LlamaContext(LlamaParams const& params, LlamaModel const& model);
  LlamaContext(LlamaContext const&) = delete;
  LlamaContext(LlamaContext&& another) noexcept;
  LlamaContext& operator=(LlamaContext const&) = delete;
  LlamaContext& operator=(LlamaContext&& another) noexcept;
  ~LlamaContext();

  bool Feed(llama_token token_pending);
  bool Feed(std::span<llama_token> tokens_pending);
  bool FeedBos();
  bool FeedEos();

  bool Eval(proto::EvalOption::Reader option);
  llama_token Predict(proto::PredictOption::Reader option);

 private:
  void MoveFrom(LlamaContext&& another) noexcept;

  size_t context_size_{};

  std::vector<llama_token> tokens_{};
  size_t tokens_begin_{};
  size_t tokens_size_{};
  size_t tokens_eval_{};

  llama_context* context_{};
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

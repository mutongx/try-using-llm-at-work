#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

#include <llama.h>

namespace muton::playground::llm {

class LlamaParams {
 public:
  LlamaParams(llama_context_params&& params);

  static LlamaParams Default();

  operator llama_context_params() const;
  llama_context_params const* operator->() const;
  llama_context_params* operator->();

 private:
  llama_context_params params_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

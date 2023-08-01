#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

#include <string>

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

 private:
  llama_context* context_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_CONTEXT_H

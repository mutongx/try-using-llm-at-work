#include "LlamaParams.h"

namespace muton::playground::llm {

LlamaParams::LlamaParams(llama_context_params&& params) : params_(params) {}

LlamaParams LlamaParams::Default() {
  return llama_context_default_params();
}

LlamaParams::operator llama_context_params() const {
  return params_;
}

llama_context_params const* LlamaParams::operator->() const {
  return &params_;
}

llama_context_params* LlamaParams::operator->() {
  return &params_;
}

}  // namespace muton::playground::llm
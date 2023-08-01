#include "LlamaContext.h"

namespace muton::playground::llm {

LlamaContext::LlamaContext(LlamaModel const& model, LlamaParams const& params)
    : context_(llama_new_context_with_model(model, params)) {}

LlamaContext::LlamaContext(LlamaContext&& another) noexcept {
  context_ = another.context_;
  another.context_ = nullptr;
}

LlamaContext& LlamaContext::operator=(LlamaContext&& another) noexcept {
  context_ = another.context_;
  another.context_ = nullptr;
  return *this;
}

LlamaContext::operator llama_context*() const {
  return context_;
}

LlamaContext::~LlamaContext() {
  if (context_ != nullptr) {
    llama_free(context_);
  }
}

}  // namespace muton::playground::llm
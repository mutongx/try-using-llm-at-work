#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

#include "llama.h"

#include "common.capnp.h"

namespace muton::playground::llm {

class LlamaParams {
 public:
  LlamaParams(proto::LlamaParams::Reader params);
  [[nodiscard]] llama_model_params const& GetModelParams() const;
  [[nodiscard]] llama_context_params const& GetContextParams() const;

 private:
  llama_model_params model_params_;
  llama_context_params context_params_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

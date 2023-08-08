#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

#include <llama.h>

#include "proto/config.capnp.h"

namespace muton::playground::llm {

class LlamaParams {
 public:
  LlamaParams(proto::LlamaParams::Reader params);
  llama_context_params Get() const;
  llama_context_params const* operator->() const;
  llama_context_params* operator->();

 private:
  llama_context_params params_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_PARAMS_H

#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_SCOPE_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_SCOPE_H

namespace muton::playground::llm {

class LlamaScope {
 public:
  LlamaScope(bool numa);
  LlamaScope(LlamaScope const&) = delete;
  LlamaScope(LlamaScope&&) = delete;
  LlamaScope& operator=(LlamaScope const&) = delete;
  LlamaScope& operator=(LlamaScope&&) = delete;
  ~LlamaScope();
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_SCOPE_H

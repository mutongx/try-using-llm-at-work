#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_MODEL_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_MODEL_H

#include <string>
#include <vector>

#include <llama.h>

#include "LlamaParams.h"

namespace muton::playground::llm {

class LlamaModel {
 public:
  LlamaModel(std::string const& path, LlamaParams const& params);
  LlamaModel(char const* path, LlamaParams const& params);
  LlamaModel(LlamaModel const&) = delete;
  LlamaModel(LlamaModel&& another) noexcept;
  LlamaModel& operator=(LlamaModel const&) = delete;
  LlamaModel& operator=(LlamaModel&& another) noexcept;
  ~LlamaModel();

  [[nodiscard]] llama_model* Get() const;

  struct Vocabulary {
    size_t size{};
    std::vector<std::string> pieces;
    std::vector<float> scores;
    std::vector<char const*> texts;
  };
  [[nodiscard]] Vocabulary GetVocabulary() const;

  [[nodiscard]] char const* GetTokenString(llama_token token) const;
  [[nodiscard]] llama_token GetBos() const;
  [[nodiscard]] llama_token GetEos() const;

 private:
  llama_model* model_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_MODEL_H

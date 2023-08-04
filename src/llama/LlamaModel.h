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
    std::vector<char const*> strings;
    std::vector<float> scores;
  };
  [[nodiscard]] Vocabulary GetVocabulary() const;

 private:
  llama_model* model_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_MODEL_H

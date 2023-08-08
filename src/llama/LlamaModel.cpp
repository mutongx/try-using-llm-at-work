#include "LlamaModel.h"

#include <llama.h>

namespace muton::playground::llm {

LlamaModel::LlamaModel(std::string const &path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path.c_str(), params.Get())) {}

LlamaModel::LlamaModel(char const *path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path, params.Get())) {}

LlamaModel::LlamaModel(LlamaModel &&another) noexcept {
  model_ = another.model_;
  another.model_ = nullptr;
}

LlamaModel &LlamaModel::operator=(LlamaModel &&another) noexcept {
  model_ = another.model_;
  another.model_ = nullptr;
  return *this;
}

llama_model *LlamaModel::Get() const {
  return model_;
}

LlamaModel::Vocabulary LlamaModel::GetVocabulary() const {
  Vocabulary result;
  result.size = llama_n_vocab_from_model(model_);
  result.strings.resize(result.size);
  result.scores.resize(result.size);
  llama_get_vocab_from_model(model_, result.strings.data(), result.scores.data(), static_cast<int>(result.size));
  return result;
}

LlamaModel::~LlamaModel() {
  if (model_ != nullptr) {
    llama_free_model(model_);
  }
}

}  // namespace muton::playground::llm
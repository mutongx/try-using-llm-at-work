#include "LlamaModel.h"

#include <llama.h>

namespace muton::playground::llm {

LlamaModel::LlamaModel(std::string const &path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path.c_str(), params)) {}

LlamaModel::LlamaModel(char const *path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path, params)) {}

LlamaModel::LlamaModel(LlamaModel &&another) noexcept {
  model_ = another.model_;
  another.model_ = nullptr;
}

LlamaModel &LlamaModel::operator=(LlamaModel &&another) noexcept {
  model_ = another.model_;
  another.model_ = nullptr;
  return *this;
}

LlamaModel::operator llama_model *() const {
  return model_;
}

LlamaModel::~LlamaModel() {
  if (model_ != nullptr) {
    llama_free_model(model_);
  }
}

}  // namespace muton::playground::llm
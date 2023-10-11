#include "LlamaModel.h"

#include <random>
#include <stdexcept>

namespace muton::playground::llm {

LlamaModel::LlamaModel(std::string const &path, const LlamaParams &params)
    : path_(path), model_(llama_load_model_from_file(path.c_str(), params.GetModelParams())) {
  if (model_ == nullptr) {
    throw std::runtime_error("failed to load model");
  }
}

LlamaModel::LlamaModel(char const *path, const LlamaParams &params)
    : path_(path), model_(llama_load_model_from_file(path, params.GetModelParams())) {
  if (model_ == nullptr) {
    throw std::runtime_error("failed to load model");
  }
}

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

LlamaVocabulary LlamaModel::GetVocabulary() const {
  return LlamaVocabulary::FromGguf(path_);
}

LlamaModel::~LlamaModel() {
  if (model_ != nullptr) {
    llama_free_model(model_);
  }
}

}  // namespace muton::playground::llm
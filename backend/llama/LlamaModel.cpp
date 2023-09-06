#include "LlamaModel.h"

#include <random>
#include <stdexcept>

#include <llama.h>

namespace muton::playground::llm {

// This struct mimics the layout of llama.cpp's llama_context, as llama.cpp didn't provide any vocabulary-related APIs
// that directly operates on llama_model. This struct can be passed to llama.cpp APIs that only uses model pointer.
// I hope that llama.cpp can add more model APIs in the future.
struct fake_llama_context {
  fake_llama_context(llama_model *model) : model(model) {}
  operator llama_context *() {
    return reinterpret_cast<llama_context *>(this);
  }
  std::mt19937 rng;
  bool has_evaluated_once{};
  int64_t t_sample_us{};
  int64_t t_eval_us{};
  int64_t t_p_eval_us{};
  int32_t n_sample{};
  int32_t n_eval{};
  int32_t n_p_eval{};
  llama_model *model;
};

LlamaModel::LlamaModel(std::string const &path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path.c_str(), params.Get())) {
  if (model_ == nullptr) {
    throw std::runtime_error("failed to load model");
  }
}

LlamaModel::LlamaModel(char const *path, const LlamaParams &params)
    : model_(llama_load_model_from_file(path, params.Get())) {
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

LlamaModel::Vocabulary LlamaModel::GetVocabulary() const {
  fake_llama_context fake_context(model_);
  Vocabulary result;
  result.size = llama_model_n_vocab(model_);
  result.pieces.resize(result.size);
  result.texts.resize(result.size);
  result.scores.resize(result.size);
  for (llama_token token{0}; token < result.size; ++token) {
    auto piece_size = -llama_token_to_piece_with_model(model_, token, nullptr, 0);
    result.pieces[token].resize(piece_size);
    llama_token_to_piece_with_model(model_, token, result.pieces[token].data(), piece_size);
    result.scores[token] = llama_token_get_score(fake_context, token);
    result.texts[token] = llama_token_get_text(fake_context, token);
  }
  return result;
}

std::string LlamaModel::GetTokenPiece(llama_token token) const {
  std::string result;
  auto piece_size = -llama_token_to_piece_with_model(model_, token, nullptr, 0);
  result.resize(piece_size);
  llama_token_to_piece_with_model(model_, token, result.data(), piece_size);
  return result;
}

char const *LlamaModel::GetTokenText(llama_token token) const {
  fake_llama_context fake_context(model_);
  return llama_token_get_text(fake_context, token);
}

llama_token LlamaModel::GetBos() const {
  fake_llama_context fake_context(model_);
  return llama_token_bos(fake_context);
}

llama_token LlamaModel::GetEos() const {
  fake_llama_context fake_context(model_);
  return llama_token_eos(fake_context);
}

LlamaModel::~LlamaModel() {
  if (model_ != nullptr) {
    llama_free_model(model_);
  }
}

}  // namespace muton::playground::llm
#include "LlamaModel.h"

#include <random>
#include <stdexcept>

namespace muton::playground::llm {

// This struct mimics the layout of llama.cpp's llama_context, as llama.cpp didn't provide any vocabulary-related APIs
// that directly operates on llama_model. This struct can be passed to llama.cpp APIs that only uses model pointer.
// I hope that llama.cpp can add more model APIs in the future.
struct fake_llama_context {
  fake_llama_context(llama_model *model) : model(*model) {
    static_cast<void>(cparams);
  }
  operator llama_context *() {
    return reinterpret_cast<llama_context *>(this);
  }
  struct {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_threads;
    uint32_t n_threads_batch;
    float rope_freq_base;
    float rope_freq_scale;
    bool mul_mat_q;
  } cparams{};
  llama_model const &model;
};

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

LlamaModel::Vocabulary LlamaModel::GetVocabulary() const {
  fake_llama_context fake_context(model_);
  Vocabulary result;
  result.size = llama_n_vocab(model_);
  result.pieces.resize(result.size);
  result.texts.resize(result.size);
  result.scores.resize(result.size);
  for (llama_token token{0}; token < result.size; ++token) {
    auto piece_size = -llama_token_to_piece(model_, token, nullptr, 0);
    result.pieces[token].resize(piece_size);
    llama_token_to_piece(model_, token, result.pieces[token].data(), piece_size);
    result.scores[token] = llama_token_get_score(fake_context, token);
    result.texts[token] = llama_token_get_text(fake_context, token);
  }
  return result;
}

std::string LlamaModel::GetTokenPiece(llama_token token) const {
  std::string result;
  auto piece_size = -llama_token_to_piece(model_, token, nullptr, 0);
  result.resize(piece_size);
  llama_token_to_piece(model_, token, result.data(), piece_size);
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
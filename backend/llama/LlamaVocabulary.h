#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_VOCABULARY_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_VOCABULARY_H

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace muton::playground::llm {

class LlamaVocabulary {
 public:
  static LlamaVocabulary FromGguf(std::string const& path);

  [[nodiscard]] enum llama_vocab_type GetType() const {
    return type_;
  }

  [[nodiscard]] size_t GetSize() const {
    return size_;
  }

  [[nodiscard]] std::string_view GetTokenText(llama_token token) {
    return tokens_text_[token];
  }

  [[nodiscard]] float GetTokenScore(llama_token token) {
    return tokens_score_[token];
  }

  [[nodiscard]] llama_token_type GetTokenType(llama_token token) {
    return tokens_type_[token];
  }

  [[nodiscard]] std::vector<std::string> const& GetMerges() {
    return merges_;
  }

  [[nodiscard]] std::string DecodeText(std::string_view text);

  [[nodiscard]] std::string GetTokenPiece(llama_token token);

 private:
  [[nodiscard]] static std::string DecodeTextSpm(std::string_view text);
  [[nodiscard]] static std::string DecodeTextBpe(std::string_view text);

  [[nodiscard]] std::string GetTokenPieceSpm(llama_token token);
  [[nodiscard]] std::string GetTokenPieceBpe(llama_token token);

  enum llama_vocab_type type_;
  size_t size_{};

  std::vector<std::string_view> tokens_text_;
  std::vector<float> tokens_score_;
  std::vector<enum llama_token_type> tokens_type_;

  std::vector<std::string> merges_;

  std::vector<char> tokens_text_store_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_VOCABULARY_H

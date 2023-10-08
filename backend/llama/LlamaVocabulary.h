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

  [[nodiscard]] size_t Size() const {
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

  [[nodiscard]] size_t GetMerge(std::string_view text, size_t split) {
    auto it = merge_ranks_.find(text);
    if (it == merge_ranks_.end()) {
      return 0;
    }
    return it->second[split];
  }

 private:

  struct string_hash {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(const char *txt) const {
      return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(std::string_view txt) const {
      return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(const std::string &txt) const {
      return std::hash<std::string>{}(txt);
    }
  };

  enum llama_vocab_type type_;
  size_t size_{};

  std::vector<std::string_view> tokens_text_;
  std::vector<float> tokens_score_;
  std::vector<enum llama_token_type> tokens_type_;

  // Keys represent merged strings, while values are ranks indexed by the length of the first part of the merged string.
  // In the associated vector, 0 indicates cannot merge, while values > 0 represent actual rank values.
  std::unordered_map<std::string, std::vector<size_t>, string_hash, std::equal_to<>> merge_ranks_;

  std::vector<char> tokens_text_store_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_VOCABULARY_H

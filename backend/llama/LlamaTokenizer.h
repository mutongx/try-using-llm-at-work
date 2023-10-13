#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H

#include <array>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "llama.h"

#include "LlamaModel.h"
#include "LlamaVocabulary.h"

#include "utilities/RegExp.h"

namespace muton::playground::llm {
class LlamaTokenizer {
 public:
  LlamaTokenizer(LlamaModel const& model);

  struct TokenizeResult {
    size_t size{};
    std::vector<llama_token> token_id;
    std::vector<size_t> token_pos;
    std::vector<size_t> token_size;
  };

  [[nodiscard]] TokenizeResult Tokenize(std::string_view text);

 private:
  // Internal struct definition

  using TokenIndex = int;
  struct Token {
    TokenIndex prev{};
    TokenIndex next{};
    std::string_view str{};
  };
  using TokenStorage = std::vector<Token>;

  struct Bigram {
    TokenIndex left{};
    TokenIndex right{};
    float score{};
    int rank{};
    std::string_view str{};
  };

  struct SpmBigramCompare {
    bool operator()(Bigram& lhs, Bigram& rhs) {
      return (lhs.score < rhs.score) || (lhs.score == rhs.score && lhs.left > rhs.left);
    }
  };
  struct BpeBigramCompare {
    bool operator()(Bigram& lhs, Bigram& rhs) {
      return (lhs.rank > rhs.rank) || (lhs.rank == rhs.rank && lhs.left > rhs.left);
    }
  };
  using SpmBigramQueue = std::priority_queue<Bigram, std::vector<Bigram>, SpmBigramCompare>;
  using BpeBigramQueue = std::priority_queue<Bigram, std::vector<Bigram>, BpeBigramCompare>;

  // Function Definition

  [[nodiscard]] static TokenStorage SeparateUTF8(std::string_view text);
  [[nodiscard]] static TokenStorage SeparateByte(std::string_view text);

  [[nodiscard]] TokenizeResult TokenizeSpm(std::string_view text);
  [[nodiscard]] TokenizeResult TokenizeBpe(std::string_view text);

  void TryAddSpmBigram(SpmBigramQueue& queue, TokenStorage const& tokens, TokenIndex left, TokenIndex right);
  void TryAddBpeBigram(BpeBigramQueue& queue, TokenStorage const& tokens, TokenIndex left, TokenIndex right);

  // Member Definition

  struct string_hash {
    using is_transparent = void;
    [[nodiscard]] size_t operator()(const char* txt) const {
      return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(std::string_view txt) const {
      return std::hash<std::string_view>{}(txt);
    }
    [[nodiscard]] size_t operator()(const std::string& txt) const {
      return std::hash<std::string>{}(txt);
    }
  };

  LlamaVocabulary vocabulary_;
  RegExp bpe_split_regex_;

  std::unordered_map<std::string, llama_token, string_hash, std::equal_to<>> pieces_mapping_;
  std::vector<llama_token> byte_token_mapping_;

  // Keys represent merged strings, while values are ranks indexed by the length of the first part of the merged string.
  // In the associated vector, 0 indicates cannot merge, while values > 0 represent actual rank values.
  std::unordered_map<std::string, std::vector<int>, string_hash, std::equal_to<>> merge_ranks_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H

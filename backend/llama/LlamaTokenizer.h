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
  using TokenIndex = int;
  struct Symbol {
    TokenIndex prev{};
    TokenIndex next{};
    std::string_view str{};
  };
  using SymbolStorage = std::vector<Symbol>;
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

  void TryAddSpmBigram(SpmBigramQueue& queue, SymbolStorage const& symbols, TokenIndex left, TokenIndex right);

  LlamaVocabulary vocabulary_;

  std::vector<std::string> pieces_;
  std::unordered_map<std::string_view, llama_token> pieces_mapping_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H

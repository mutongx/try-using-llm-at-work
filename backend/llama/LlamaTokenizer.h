#ifndef MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H
#define MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H

#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <llama.h>

#include "LlamaModel.h"

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
  TokenizeResult Tokenize(std::string_view text);

 private:
  static std::array<size_t, 16> Utf8SymbolSizeLookupTable;

  using SentencePieceIndex = int;
  struct SentencePieceSymbol {
    SentencePieceIndex prev{};
    SentencePieceIndex next{};
    const char* ptr{};
    size_t size{};
  };
  using SentencePieceSymbolStorage = std::vector<SentencePieceSymbol>;
  struct SentencePieceBigram {
    SentencePieceIndex left{};
    SentencePieceIndex right{};
    float score{};
    size_t size{};
  };
  struct SentencePieceBigramCompare {
    bool operator()(SentencePieceBigram& lhs, SentencePieceBigram& rhs) {
      return (lhs.score < rhs.score) || (lhs.score == rhs.score && lhs.left > rhs.left);
    }
  };
  using SentencePieceBigramQueue =
      std::priority_queue<SentencePieceBigram, std::vector<SentencePieceBigram>, SentencePieceBigramCompare>;

  void TryAddBigram(SentencePieceBigramQueue& queue,
                    SentencePieceSymbolStorage const& symbols,
                    SentencePieceIndex left,
                    SentencePieceIndex right);

  std::vector<std::string> strings_;
  std::vector<float> scores_;
  std::unordered_map<std::string_view, llama_token> mapping_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_LLAMA_LLAMA_TOKENIZER_H

#include "LlamaTokenizer.h"

#include <string_view>

namespace muton::playground::llm {

std::array<size_t, 16> LlamaTokenizer::Utf8SymbolSizeLookupTable = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};

LlamaTokenizer::LlamaTokenizer(const LlamaModel& model) : vocabulary_(model.GetVocabulary()) {
  pieces_.reserve(vocabulary_.GetSize());
  for (int i = 0; i < vocabulary_.GetSize(); ++i) {
    pieces_.push_back(vocabulary_.GetTokenPiece(i));
    pieces_mapping_[pieces_[i]] = i;
  }
}

LlamaTokenizer::TokenizeResult LlamaTokenizer::Tokenize(std::string_view text) {
  TokenizeResult result;
  SentencePieceBigramQueue queue;
  SentencePieceSymbolStorage symbols;
  SentencePieceIndex index{0};
  size_t offset{0};
  while (offset < text.size()) {
    auto& item = symbols.emplace_back();
    item.ptr = text.data() + offset;
    item.size = std::min(text.size() - offset, Utf8SymbolSizeLookupTable[static_cast<uint8_t>(text[offset]) >> 4]);
    item.prev = index - 1;
    item.next = index + 1;
    offset += item.size;
    ++index;
  }
  if (!symbols.empty()) {
    symbols.back().next = -1;
  }
  for (SentencePieceIndex i = 1; i < symbols.size(); ++i) {
    TryAddBigram(queue, symbols, i - 1, i);
  }
  while (!queue.empty()) {
    auto item = queue.top();
    queue.pop();
    auto& sym_left = symbols[item.left];
    auto& sym_right = symbols[item.right];
    if (sym_left.size == 0 || sym_right.size == 0 || (sym_left.size + sym_right.size != item.size)) {
      continue;
    }
    sym_left.size += sym_right.size;
    sym_right.size = 0;
    sym_left.next = sym_right.next;
    if (sym_right.next >= 0) {
      symbols[sym_right.next].prev = item.left;
    }
    TryAddBigram(queue, symbols, sym_left.prev, item.left);
    TryAddBigram(queue, symbols, item.left, sym_left.next);
  }
  if (!symbols.empty()) {
    for (index = 0; index != -1; index = symbols[index].next) {
      auto& item = symbols[index];
      std::string_view symbol{item.ptr, item.size};
      auto token_it = pieces_mapping_.find(symbol);
      if (token_it == pieces_mapping_.end()) {
        for (size_t i = 0; i < symbol.size(); ++i) {
          auto token_id = pieces_mapping_.at(std::string_view(symbol.data() + i, 1));
          result.token_id.push_back(token_id);
          result.token_pos.push_back(symbol.data() + i - text.data());
          result.token_size.push_back(1);
          result.size += 1;
        }
      } else {
        result.token_id.push_back(token_it->second);
        result.token_pos.push_back(item.ptr - text.data());
        result.token_size.push_back(item.size);
        result.size += 1;
      }
    }
  }
  return result;
}

void LlamaTokenizer::TryAddBigram(SentencePieceBigramQueue& queue,
                                  SentencePieceSymbolStorage const& symbols,
                                  SentencePieceIndex left,
                                  SentencePieceIndex right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto symbol = std::string_view(symbols[left].ptr, symbols[left].size + symbols[right].size);
  auto token_it = pieces_mapping_.find(symbol);
  if (token_it == pieces_mapping_.end()) {
    return;
  }
  queue.emplace(SentencePieceBigram{left, right, vocabulary_.GetTokenScore(token_it->second), symbol.size()});
}

}  // namespace muton::playground::llm

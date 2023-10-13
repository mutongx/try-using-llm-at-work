#include "LlamaTokenizer.h"

#include <string_view>

#include "utilities/UTF8Text.h"

namespace muton::playground::llm {

LlamaTokenizer::LlamaTokenizer(const LlamaModel& model) : vocabulary_(model.GetVocabulary()) {
  pieces_.reserve(vocabulary_.GetSize());
  for (int i = 0; i < vocabulary_.GetSize(); ++i) {
    pieces_.push_back(vocabulary_.GetTokenPiece(i));
    pieces_mapping_[pieces_[i]] = i;
  }
  if (vocabulary_.GetType() == LLAMA_VOCAB_TYPE_BPE) {
    int rank{0};
    for (auto const& merge : vocabulary_.GetMerges()) {
      size_t space = merge.find(' ');
      if (space == 0 || space == std::string::npos) {
        throw std::runtime_error(std::string().append("invalid merge specification: ").append(merge));
      }
      std::string key;
      key.append(vocabulary_.DecodeText(std::string_view(merge.data(), space)));
      size_t split{key.size()};
      key.append(vocabulary_.DecodeText(std::string_view(merge.data() + space + 1)));
      merge_ranks_.try_emplace(key, key.size()).first->second[split] = ++rank;
    }
  }
}

LlamaTokenizer::TokenizeResult LlamaTokenizer::Tokenize(std::string_view text) {
  TokenizeResult result;
  SpmBigramQueue queue;
  SymbolStorage symbols;
  TokenIndex index{0};
  for (auto symbol : UTF8Text(text)) {
    auto& item = symbols.emplace_back();
    item.prev = index - 1;
    item.next = index + 1;
    item.str = symbol.str;
    ++index;
  }
  if (!symbols.empty()) {
    symbols.back().next = -1;
  }
  for (TokenIndex i = 1; i < symbols.size(); ++i) {
    TryAddSpmBigram(queue, symbols, i - 1, i);
  }
  while (!queue.empty()) {
    auto item = queue.top();
    queue.pop();
    auto& sym_left = symbols[item.left];
    auto& sym_right = symbols[item.right];
    if (sym_left.str.empty() || sym_right.str.empty() ||
        (sym_left.str.size() + sym_right.str.size() != item.str.size())) {
      continue;
    }
    sym_left.str = std::string_view(sym_left.str.data(), sym_left.str.size() + sym_right.str.size());
    sym_right.str = std::string_view("");
    sym_left.next = sym_right.next;
    if (sym_right.next >= 0) {
      symbols[sym_right.next].prev = item.left;
    }
    TryAddSpmBigram(queue, symbols, sym_left.prev, item.left);
    TryAddSpmBigram(queue, symbols, item.left, sym_left.next);
  }
  if (!symbols.empty()) {
    for (index = 0; index != -1; index = symbols[index].next) {
      auto& item = symbols[index];
      auto token_it = pieces_mapping_.find(item.str);
      if (token_it == pieces_mapping_.end()) {
        for (size_t i = 0; i < item.str.size(); ++i) {
          auto token_id = pieces_mapping_.at(std::string_view(item.str.data() + i, 1));
          result.token_id.push_back(token_id);
          result.token_pos.push_back(item.str.data() + i - text.data());
          result.token_size.push_back(1);
          result.size += 1;
        }
      } else {
        result.token_id.push_back(token_it->second);
        result.token_pos.push_back(item.str.data() - text.data());
        result.token_size.push_back(item.str.size());
        result.size += 1;
      }
    }
  }
  return result;
}

void LlamaTokenizer::TryAddSpmBigram(SpmBigramQueue& queue,
                                     SymbolStorage const& symbols,
                                     TokenIndex left,
                                     TokenIndex right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto symbol = std::string_view(symbols[left].str.data(), symbols[left].str.size() + symbols[right].str.size());
  auto token_it = pieces_mapping_.find(symbol);
  if (token_it == pieces_mapping_.end()) {
    return;
  }
  auto score = vocabulary_.GetTokenScore(token_it->second);
  queue.emplace(Bigram{.left = left, .right = right, .score = score, .str = symbol});
}

}  // namespace muton::playground::llm

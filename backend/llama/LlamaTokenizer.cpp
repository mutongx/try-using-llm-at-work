#include "LlamaTokenizer.h"

#include <stdexcept>
#include <string_view>

#include "utilities/RegExp.h"
#include "utilities/UTF8Text.h"

namespace muton::playground::llm {

static auto const* bpe_split_pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

LlamaTokenizer::LlamaTokenizer(const LlamaModel& model)
    : vocabulary_(model.GetVocabulary()), bpe_split_regex_(bpe_split_pattern) {
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
  if (vocabulary_.GetType() == LLAMA_VOCAB_TYPE_SPM) {
    return TokenizeSpm(text);
  }
  if (vocabulary_.GetType() == LLAMA_VOCAB_TYPE_BPE) {
    return TokenizeBpe(text);
  }
  throw std::runtime_error("invalid vocabulary type");
}

LlamaTokenizer::TokenizeResult LlamaTokenizer::TokenizeSpm(std::string_view text) {
  TokenizeResult result;
  TokenStorage tokens;
  SpmBigramQueue queue;
  {
    TokenIndex index{0};
    for (auto symbol : UTF8Text(text)) {
      auto& item = tokens.emplace_back();
      item.prev = index - 1;
      item.next = index + 1;
      item.str = symbol.str;
      ++index;
    }
  }
  if (!tokens.empty()) {
    tokens.back().next = -1;
  }
  for (TokenIndex i = 1; i < tokens.size(); ++i) {
    TryAddSpmBigram(queue, tokens, i - 1, i);
  }
  while (!queue.empty()) {
    auto item = queue.top();
    queue.pop();
    auto& sym_left = tokens[item.left];
    auto& sym_right = tokens[item.right];
    if (sym_left.str.empty() || sym_right.str.empty() ||
        (sym_left.str.size() + sym_right.str.size() != item.str.size())) {
      continue;
    }
    if (sym_left.str.data() + sym_left.str.size() != sym_right.str.data()) {
      throw std::runtime_error("something is wrong in token merge");
    }
    sym_left.str = std::string_view(sym_left.str.data(), sym_left.str.size() + sym_right.str.size());
    sym_right.str = std::string_view("");
    sym_left.next = sym_right.next;
    if (sym_right.next >= 0) {
      tokens[sym_right.next].prev = item.left;
    }
    TryAddSpmBigram(queue, tokens, sym_left.prev, item.left);
    TryAddSpmBigram(queue, tokens, item.left, sym_left.next);
  }
  if (!tokens.empty()) {
    for (TokenIndex index = 0; index != -1; index = tokens[index].next) {
      auto const& item = tokens[index];
      if (item.str.empty()) {
        continue;
      }
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

LlamaTokenizer::TokenizeResult LlamaTokenizer::TokenizeBpe(std::string_view text) {
  TokenizeResult result;
  std::vector<TokenStorage> all_tokens;
  for (auto component : bpe_split_regex_.Match(text)) {
    TokenStorage& tokens{all_tokens.emplace_back()};
    BpeBigramQueue queue;
    for (TokenIndex index = 0; index < component.size(); ++index) {
      auto& item = tokens.emplace_back();
      item.prev = index - 1;
      item.next = index + 1;
      item.str = std::string_view(component.data() + index, 1);
    }
    if (!tokens.empty()) {
      tokens.back().next = -1;
    }
    for (TokenIndex i = 1; i < tokens.size(); ++i) {
      TryAddBpeBigram(queue, tokens, i - 1, i);
    }
    while (!queue.empty()) {
      auto item = queue.top();
      queue.pop();
      auto& sym_left = tokens[item.left];
      auto& sym_right = tokens[item.right];
      if (sym_left.str.empty() || sym_right.str.empty() ||
          (sym_left.str.size() + sym_right.str.size() != item.str.size())) {
        continue;
      }
      if (sym_left.str.data() + sym_left.str.size() != sym_right.str.data()) {
        throw std::runtime_error("something is wrong in token merge");
      }
      sym_left.str = std::string_view(sym_left.str.data(), sym_left.str.size() + sym_right.str.size());
      sym_right.str = std::string_view("");
      sym_left.next = sym_right.next;
      if (sym_right.next >= 0) {
        tokens[sym_right.next].prev = item.left;
      }
      TryAddBpeBigram(queue, tokens, sym_left.prev, item.left);
      TryAddBpeBigram(queue, tokens, item.left, sym_left.next);
    }
  }
  for (auto const& tokens : all_tokens) {
    if (tokens.empty()) {
      continue;
    }
    for (TokenIndex index = 0; index != -1; index = tokens[index].next) {
      auto const& item = tokens[index];
      if (item.str.empty()) {
        continue;
      }
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
                                     TokenStorage const& tokens,
                                     TokenIndex left,
                                     TokenIndex right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto str = std::string_view(tokens[left].str.data(), tokens[left].str.size() + tokens[right].str.size());
  auto token_it = pieces_mapping_.find(str);
  if (token_it == pieces_mapping_.end()) {
    return;
  }
  auto score = vocabulary_.GetTokenScore(token_it->second);
  queue.emplace(Bigram{.left = left, .right = right, .score = score, .str = str});
}

void LlamaTokenizer::TryAddBpeBigram(BpeBigramQueue& queue,
                                     TokenStorage const& tokens,
                                     TokenIndex left,
                                     TokenIndex right) {
  if (left == -1 || right == -1) {
    return;
  }
  auto str = std::string_view(tokens[left].str.data(), tokens[left].str.size() + tokens[right].str.size());
  auto split = tokens[left].str.size();
  int rank = 0;
  auto merge_it = merge_ranks_.find(str);
  if (merge_it != merge_ranks_.end()) {
    rank = merge_it->second[split];
  }
  if (rank == 0) {
    return;
  }
  queue.emplace(Bigram{.left = left, .right = right, .rank = rank, .str = str});
}

}  // namespace muton::playground::llm

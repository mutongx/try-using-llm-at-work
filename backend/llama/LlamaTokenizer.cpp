#include "LlamaTokenizer.h"

#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

#include "utilities/RegExp.h"
#include "utilities/UTF8Text.h"

namespace muton::playground::llm {

static auto const* bpe_split_pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

LlamaTokenizer::LlamaTokenizer(const LlamaModel& model)
    : vocabulary_(model.GetVocabulary()), bpe_split_regex_(bpe_split_pattern) {
  // Create a piece -> token and byte -> token mapping
  auto vocab_type = vocabulary_.GetType();
  byte_token_mapping_.resize(256, -1);
  for (int i = 0; i < vocabulary_.GetSize(); ++i) {
    auto type = vocabulary_.GetTokenType(i);
    auto piece = vocabulary_.GetTokenPiece(i);
    if (type == LLAMA_TOKEN_TYPE_NORMAL) {
      auto emplace = pieces_mapping_.try_emplace(std::move(piece), i);
      if (!emplace.second) {
        throw std::runtime_error(fmt::format("duplicate token piece: {}/{}", emplace.first->second, i));
      }
      // For BPE tokenizer, we also use piece as byte token
      if (vocab_type == LLAMA_VOCAB_TYPE_BPE && piece.size() == 1) {
        auto byte = static_cast<uint8_t>(piece[0]);
        byte_token_mapping_[byte] = i;
      }
    } else if (type == LLAMA_TOKEN_TYPE_BYTE) {
      if (vocab_type == LLAMA_VOCAB_TYPE_BPE) {
        throw std::runtime_error("unexpected byte token in BPE vocabulary");
      }
      auto byte = static_cast<uint8_t>(piece[0]);
      byte_token_mapping_[byte] = i;
    }
  }
  // Create a merge -> merge pos -> rank mapping
  if (vocab_type == LLAMA_VOCAB_TYPE_BPE) {
    int rank{0};
    for (auto const& merge : vocabulary_.GetMerges()) {
      size_t space = merge.find(' ');
      if (space == 0 || space == std::string::npos) {
        throw std::runtime_error(fmt::format("invalid merge specification: {}", merge));
      }
      // Remove the space, and decode byte representation to bytes
      std::string key;
      key.reserve(merge.size());
      key.append(vocabulary_.DecodeText(std::string_view(merge.data(), space)));
      size_t split{key.size()};
      key.append(vocabulary_.DecodeText(std::string_view(merge.data() + space + 1)));
      size_t total{key.size()};
      // Use merges_ as a storage, and use string_view as key in merge_ranks_
      merge_ranks_.try_emplace(std::move(key), total).first->second[split] = ++rank;
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

LlamaTokenizer::TokenStorage LlamaTokenizer::SeparateUTF8(std::string_view text) {
  TokenStorage result;
  TokenIndex index{0};
  for (auto symbol : UTF8Text(text)) {
    auto& item = result.emplace_back();
    item.prev = index - 1;
    item.next = index + 1;
    item.str = symbol.str;
    ++index;
  }
  if (!result.empty()) {
    result.back().next = -1;
  }
  return result;
}

LlamaTokenizer::TokenStorage LlamaTokenizer::SeparateByte(std::string_view text) {
  TokenStorage result;
  for (TokenIndex index = 0; index < text.size(); ++index) {
    auto& item = result.emplace_back();
    item.prev = index - 1;
    item.next = index + 1;
    item.str = std::string_view(text.data() + index, 1);
  }
  if (!result.empty()) {
    result.back().next = -1;
  }
  return result;
}

LlamaTokenizer::TokenizeResult LlamaTokenizer::TokenizeSpm(std::string_view text) {
  TokenizeResult result;
  TokenStorage tokens{SeparateUTF8(text)};
  SpmBigramQueue queue;
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
          auto byte = static_cast<uint8_t>(item.str[i]);
          auto token_id = byte_token_mapping_[byte];
          if (token_id == -1) {
            throw std::runtime_error(fmt::format("cannot find byte token for {}", byte));
          }
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
    TokenStorage tokens{SeparateByte(component)};
    BpeBigramQueue queue;
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
    all_tokens.emplace_back(std::move(tokens));
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
          auto byte = static_cast<uint8_t>(item.str[i]);
          auto token_id = byte_token_mapping_[byte];
          if (token_id == -1) {
            throw std::runtime_error(fmt::format("cannot find byte token for {}", byte));
          }
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

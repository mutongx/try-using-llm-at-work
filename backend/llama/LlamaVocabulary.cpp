#include "LlamaVocabulary.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "ggml.h"

#include "utilities/UTF8Text.h"

namespace muton::playground::llm {

LlamaVocabulary LlamaVocabulary::FromGguf(std::string const& path) {
  class GgufWrapper {
   public:
    GgufWrapper(std::string const& path) {
      ctx_ = gguf_init_from_file(path.data(), {.no_alloc = true, .ctx = nullptr});
      if (ctx_ == nullptr) {
        throw std::runtime_error("failed to load model file");
      }
      n_kv_ = gguf_get_n_kv(ctx_);
      for (int i = 0; i < n_kv_; ++i) {
        kv_map_[gguf_get_key(ctx_, i)] = i;
      }
    }
    GgufWrapper(GgufWrapper const&) = delete;
    GgufWrapper(GgufWrapper&&) = delete;
    GgufWrapper& operator=(GgufWrapper const&) = delete;
    GgufWrapper& operator=(GgufWrapper&&) = delete;
    ~GgufWrapper() {
      gguf_free(ctx_);
    }
    operator gguf_context*() {
      return ctx_;
    }
    int GetKeyId(std::string const& key,
                 gguf_type expected_type = static_cast<gguf_type>(-1),
                 gguf_type expected_array_type = static_cast<gguf_type>(-1)) {
      auto it = kv_map_.find(key);
      if (it == kv_map_.end()) {
        return -1;
      }
      if (expected_type != -1) {
        if (gguf_get_kv_type(ctx_, it->second) != expected_type) {
          return -2;
        }
      }
      if (expected_array_type != -1) {
        if (gguf_get_arr_type(ctx_, it->second) != expected_array_type) {
          return -3;
        }
      }
      return it->second;
    }
    int MustGetKeyId(std::string const& key,
                     gguf_type expected_type = static_cast<gguf_type>(-1),
                     gguf_type expected_array_type = static_cast<gguf_type>(-1)) {
      auto id = GetKeyId(key, expected_type, expected_array_type);
      if (id == -1) {
        throw std::runtime_error(std::string().append("Key ").append(key).append(" not found"));
      }
      if (id == -2) {
        throw std::runtime_error(std::string().append("Key ").append(key).append(" type mismatch"));
      }
      if (id == -3) {
        throw std::runtime_error(std::string().append("Key ").append(key).append(" array type mismatch"));
      }
      return id;
    }

   private:
    gguf_context* ctx_{};
    int n_kv_{};
    std::unordered_map<std::string, int> kv_map_;
  };

  LlamaVocabulary result;
  GgufWrapper ctx(path);

  auto model_id = ctx.MustGetKeyId("tokenizer.ggml.model", GGUF_TYPE_STRING);
  char const* model_str = gguf_get_val_str(ctx, model_id);
  if (strcmp(model_str, "llama") == 0) {
    result.type_ = llama_vocab_type::LLAMA_VOCAB_TYPE_SPM;
  } else if (strcmp(model_str, "gpt2") == 0) {
    result.type_ = llama_vocab_type::LLAMA_VOCAB_TYPE_BPE;
  } else {
    throw std::runtime_error("unknown model type");
  }

  auto tokens_id = ctx.MustGetKeyId("tokenizer.ggml.tokens", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING);
  auto scores_id = ctx.GetKeyId("tokenizer.ggml.scores", GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32);
  auto token_type_id = ctx.GetKeyId("tokenizer.ggml.token_type", GGUF_TYPE_ARRAY, GGUF_TYPE_INT32);

  auto vocab_size = gguf_get_arr_n(ctx, tokens_id);

  // Prepare for tokens storage
  std::vector<size_t> tokens_strlen(vocab_size);
  size_t tokens_store_size{0};
  for (size_t i = 0; i < vocab_size; ++i) {
    tokens_strlen[i] = strlen(gguf_get_arr_str(ctx, tokens_id, static_cast<int>(i)));
    tokens_store_size += tokens_strlen[i] + 1;
  }

  // Extract scores and token type store
  float const* scores_arr =
      scores_id >= 0 ? reinterpret_cast<float const*>(gguf_get_arr_data(ctx, scores_id)) : nullptr;
  int32_t const* token_type_arr =
      token_type_id >= 0 ? reinterpret_cast<int32_t const*>(gguf_get_arr_data(ctx, token_type_id)) : nullptr;

  // Assign results
  result.size_ = vocab_size;
  result.tokens_text_.resize(vocab_size);
  result.tokens_score_.resize(vocab_size);
  result.tokens_type_.resize(vocab_size);
  result.tokens_text_store_.resize(tokens_store_size, 0);

  // Then assign all array values
  size_t tokens_store_offset{0};
  for (size_t i = 0; i < vocab_size; ++i) {
    memcpy(result.tokens_text_store_.data() + tokens_store_offset,
           gguf_get_arr_str(ctx, tokens_id, static_cast<int>(i)),
           tokens_strlen[i]);
    result.tokens_text_[i] = std::string_view(result.tokens_text_store_.data() + tokens_store_offset, tokens_strlen[i]);
    result.tokens_score_[i] = scores_arr != nullptr ? scores_arr[i] : 0.0F;
    result.tokens_type_[i] =
        token_type_arr != nullptr ? static_cast<llama_token_type>(token_type_arr[i]) : LLAMA_TOKEN_TYPE_NORMAL;
    tokens_store_offset += tokens_strlen[i] + 1;
  }

  // For GPT2 tokenizers, we need a merges array
  if (strcmp(model_str, "gpt2") == 0) {
    auto merges_id = ctx.MustGetKeyId("tokenizer.ggml.merges", GGUF_TYPE_ARRAY, GGUF_TYPE_STRING);
    if (merges_id >= 0) {
      auto merges_size = gguf_get_arr_n(ctx, merges_id);
      for (size_t i = 0; i < merges_size; ++i) {
        char const* merge = gguf_get_arr_str(ctx, merges_id, static_cast<int>(i));
        char const* space = strchr(merge + 1, ' ');
        if (space == nullptr) {
          throw std::runtime_error(std::string().append("invalid merge specification: ").append(merge));
        }
        std::string key;
        size_t total = strlen(merge) - 1;
        size_t split = space - merge;
        key.resize(total);
        memcpy(key.data(), merge, split);
        memcpy(key.data() + split, space + 1, total - split);
        result.merge_ranks_.try_emplace(key, key.size()).first->second[split] = i + 1;
      }
    }
  }

  return result;
}

std::string LlamaVocabulary::GetTokenPieceSpm(llama_token token) {
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_NORMAL) {
    std::string result;
    auto const* p = tokens_text_[token].data();
    while (*p != 0) {
      // Space is unicode LOWER ONE EIGHTH BLOCK
      if (*p == '\xe2' && *(p + 1) != 0 && *(p + 1) == '\x96' && *(p + 2) != 0 && *(p + 2) == '\x81') {
        result.push_back(' ');
        p += 3;
      } else {
        result.push_back(*p);
        ++p;
      }
    }
    return result;
  }
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_UNKNOWN) {
    return "\xe2\x96\x85";
  }
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_CONTROL) {
    return "";
  }
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_BYTE) {
    std::string result(1, 0);
    // Parse <0xXX> to actual byte
    auto const* p = tokens_text_[token].data();
    if (p[0] != '<' || p[1] != '0' || p[2] != 'x') {
      throw std::runtime_error("invalid byte token text");
    }
    for (p += 3; *p != '>' && *p != 0; ++p) {
      result[0] *= 16;
      if ('0' <= *p && *p <= '9') {
        result[0] += (*p - '0');
      } else if ('A' <= *p && *p <= 'F') {
        result[0] += (*p - 'A' + 10);
      } else {
        throw std::runtime_error("invalid byte token text");
      }
    }
    if (*p != '>') {
      throw std::runtime_error("invalid byte token text");
    }
    return result;
  }
  return "";
}

std::string LlamaVocabulary::GetTokenPieceBpe(llama_token token) {
  class BpeByteMap {
   public:
    BpeByteMap() : byte_map_(512, 0), set_mask_(512, 0) {
      size_t invisible{0};
      uint8_t current{0};
      do {
        // See https://github.com/karpathy/minGPT/blob/master/mingpt/bpe.py for detailed explanations
        // ChatGPT: In summary, OpenAI maps every byte from 0 to 255 to Unicode characters, some of which are preserved
        // while others are shifted to new characters to ensure they appear visually pleasing in the final dictionary.
        if (33 <= current && current <= 126 || 161 <= current && current <= 172 || 174 <= current && current <= 255) {
          byte_map_[current] = current;
          set_mask_[current] = 1;
        } else {
          byte_map_[invisible + 256] = current;
          set_mask_[invisible + 256] = 1;
          ++invisible;
        }
      } while (current++ != 255);
    }
    char Get(uint32_t cp) {
      if (set_mask_[cp] == 0) {
        throw std::runtime_error("invalid code page");
      }
      return static_cast<char>(byte_map_[cp]);
    }

   private:
    std::vector<uint8_t> byte_map_;
    std::vector<uint8_t> set_mask_;
  };
  static BpeByteMap byte_map;
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_NORMAL) {
    std::string result;
    for (auto symbol : UTF8Text(tokens_text_[token])) {
      result.push_back(byte_map.Get(symbol.cp));
    }
    return result;
  }
  if (tokens_type_[token] == LLAMA_TOKEN_TYPE_CONTROL) {
    return "";
  }
  return "";
}

}  // namespace muton::playground::llm
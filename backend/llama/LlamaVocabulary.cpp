#include "LlamaVocabulary.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "ggml.h"

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

}  // namespace muton::playground::llm
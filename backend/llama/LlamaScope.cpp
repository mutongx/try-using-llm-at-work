#include "LlamaScope.h"

#include <llama.h>

namespace muton::playground::llm {

LlamaScope::LlamaScope(bool numa) {
  llama_backend_init(numa);
}

LlamaScope::~LlamaScope() {
  llama_backend_free();
}

}  // namespace muton::playground::llm
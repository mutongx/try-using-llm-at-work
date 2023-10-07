#ifndef MUTON_PLAYGROUND_LLM_INITIALIZE_H
#define MUTON_PLAYGROUND_LLM_INITIALIZE_H

#include <vector>

#include "llama.h"
#include "oniguruma.h"

namespace muton::playground::llm {

struct InitializeOptions {
  bool llama_numa{false};
  std::vector<OnigEncoding> onig_encodings{&OnigEncodingUTF8};
};

void Initialize(InitializeOptions options);

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_INITIALIZE_H

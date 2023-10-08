#include "catch2/catch_test_macros.hpp"

#include "config/Config.h"
#include "llama/LlamaContext.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaVocabulary.h"

TEST_CASE("The vocabulary loads correctly", "[llama][vocabulary]") {
  auto config = muton::playground::llm::Config::Read("config-test.json");
  auto vocab = muton::playground::llm::LlamaVocabulary::FromGguf(config->getModel().cStr());

  // Construct model using original llama.cpp
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  muton::playground::llm::LlamaContext context(params, model);

  // Compare vocabulary with llama.cpp
  REQUIRE(vocab.Size() == llama_n_vocab(model.Get()));
  for (size_t i = 0; i < vocab.Size(); ++i) {
    REQUIRE(vocab.GetTokenText(i) == llama_token_get_text(context.Get(), i));
    REQUIRE(vocab.GetTokenScore(i) == llama_token_get_score(context.Get(), i));
    REQUIRE(vocab.GetTokenType(i) == llama_token_get_type(context.Get(), i));
  }

}

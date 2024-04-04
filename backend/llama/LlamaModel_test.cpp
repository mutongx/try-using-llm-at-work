#include "catch2/catch_test_macros.hpp"

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"

// Unit tests for LlamaModel

TEST_CASE("The model loads correctly", "[llama][model]") {
  // Build the model (llama-2-7b.Q4_0.gguf).
  auto config = muton::playground::llm::Config::Read("config-test.json");
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  REQUIRE(model.Get() != nullptr);
}

TEST_CASE("The model can handle BOS and EOS token correctly", "[llama][model]") {
  auto config = muton::playground::llm::Config::Read("config-test.json");
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  muton::playground::llm::LlamaVocabulary vocabulary{model.GetVocabulary()};
  SECTION("The model gets the begin of stream token correctly") {
    // Test cases for LlamaModel::GetBos():
    // 1. The begin of stream token is <s>.
    REQUIRE(model.GetBos() == 1);
    REQUIRE(vocabulary.GetTokenText(model.GetBos()) == "<s>");
  }
  SECTION("The model gets the end of stream token correctly") {
    // Test cases for LlamaModel::GetEos():
    // 1. The end of stream token is </s>.
    REQUIRE(model.GetEos() == 2);
    REQUIRE(vocabulary.GetTokenText(model.GetEos()) == "</s>");
  }
}

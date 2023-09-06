#include "catch2/catch_test_macros.hpp"

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"

// Unit tests for LlamaModel

TEST_CASE("The model loads correctly", "[llama][model]") {
  // Build the model (llama-2-7b.Q4_0.gguf).
  auto config = muton::playground::llm::Config::Read();
  muton::playground::llm::LlamaScope scope(true);
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  auto vocab = model.GetVocabulary();
  SECTION("The model gets vocabulary correctly") {
    // Test cases for LlamaModel::GetVocabulary():
    // 1. The vocabulary size is 32000, that is, vocab.size == vocab.strings.size == vocab.scores.size == 32000.
    REQUIRE(vocab.size == 32000);
    REQUIRE(vocab.size == vocab.pieces.size());
    REQUIRE(vocab.size == vocab.scores.size());
    REQUIRE(vocab.size == vocab.texts.size());
    // 2. The 0 token is <unk> with score 0.
    REQUIRE(strcmp(vocab.texts[0], "<unk>") == 0);
    REQUIRE(vocab.scores[0] == 0);
    // 3. The 29902nd token is I with score -29643.
    REQUIRE(strcmp(vocab.texts[29902], "I") == 0);
    REQUIRE(vocab.scores[29902] == -29643);
    // 4. The 5031st token is pat with score -4772.
    REQUIRE(strcmp(vocab.texts[5031], "pat") == 0);
    REQUIRE(vocab.scores[5031] == -4772);
    // 5. The 8767th token is ubuntu with score -8508.
    REQUIRE(strcmp(vocab.texts[8767], "ubuntu") == 0);
    REQUIRE(vocab.scores[8767] == -8508);
  }
  SECTION("The model gets string from token correctly") {
    // Test cases for LlamaModel::GetTokenString():
    // 1. The 0 token is <unk>.
    REQUIRE(strcmp(model.GetTokenString(0), "<unk>") == 0);
    // 2. The 29902nd token is I.
    REQUIRE(strcmp(model.GetTokenString(29902), "I") == 0);
    // 3. The 5031st token is pat.
    REQUIRE(strcmp(model.GetTokenString(5031), "pat") == 0);
    // 4. The 8767th token is ubuntu.
    REQUIRE(strcmp(model.GetTokenString(8767), "ubuntu") == 0);
  }
  SECTION("The model gets the begin of stream token correctly") {
    // Test cases for LlamaModel::GetBos():
    // 1. The begin of stream token is <s>.
    REQUIRE(model.GetBos() == 1);
    REQUIRE(strcmp(model.GetTokenString(model.GetBos()), "<s>") == 0);
  }
  SECTION("The model gets the end of stream token correctly") {
    // Test cases for LlamaModel::GetEos():
    // 1. The end of stream token is </s>.
    REQUIRE(model.GetEos() == 2);
    REQUIRE(strcmp(model.GetTokenString(model.GetEos()), "</s>") == 0);
  }
}

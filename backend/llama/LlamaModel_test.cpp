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
  SECTION("The model gets string from token correctly") {
    // Test cases for LlamaModel::GetTokenText():
    // 1. The 0 token is <unk>.
    REQUIRE(strcmp(model.GetTokenText(0), "<unk>") == 0);
    // 2. The 29902nd token is I.
    REQUIRE(strcmp(model.GetTokenText(29902), "I") == 0);
    // 3. The 5031st token is pat.
    REQUIRE(strcmp(model.GetTokenText(5031), "pat") == 0);
    // 4. The 8767th token is ubuntu.
    REQUIRE(strcmp(model.GetTokenText(8767), "ubuntu") == 0);
  }
  SECTION("The model gets piece from token correctly") {
    // Test cases for LlamaModel::GetTokenPiece():
    // 1. The 29902nd token is I.
    REQUIRE(model.GetTokenPiece(29902) == "I");
    // 2. The 5031st token is pat.
    REQUIRE(model.GetTokenPiece(5031) == "pat");
    // 3. The 8767th token is ubuntu.
    REQUIRE(model.GetTokenPiece(8767) == "ubuntu");
  }
  SECTION("The model gets the begin of stream token correctly") {
    // Test cases for LlamaModel::GetBos():
    // 1. The begin of stream token is <s>.
    REQUIRE(model.GetBos() == 1);
    REQUIRE(strcmp(model.GetTokenText(model.GetBos()), "<s>") == 0);
  }
  SECTION("The model gets the end of stream token correctly") {
    // Test cases for LlamaModel::GetEos():
    // 1. The end of stream token is </s>.
    REQUIRE(model.GetEos() == 2);
    REQUIRE(strcmp(model.GetTokenText(model.GetEos()), "</s>") == 0);
  }
}

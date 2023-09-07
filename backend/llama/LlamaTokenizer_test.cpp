#include "catch2/catch_test_macros.hpp"

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaScope.h"
#include "llama/LlamaTokenizer.h"

TEST_CASE("The model tokenizes a string correctly", "[llama][tokenizer]") {
  // Build the model (llama-2-7b.Q4_0.gguf).
  auto config = muton::playground::llm::Config::Read();
  muton::playground::llm::LlamaScope scope(true);
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  muton::playground::llm::LlamaTokenizer tokenizer(model);
  auto vocab = model.GetVocabulary();
  // Test cases for LlamaTokenizer::Tokenize():
  SECTION("The model tokenizes English sentence correctly") {
    auto result = tokenizer.Tokenize("I pat ubuntu");
    // 1. The result has 3 tokens.
    REQUIRE(result.size == 3);
    REQUIRE(result.size == result.token_id.size());
    REQUIRE(result.size == result.token_pos.size());
    REQUIRE(result.size == result.token_size.size());
    // 2. The first token is "I".
    REQUIRE(model.GetTokenPiece(result.token_id[0]) == "I");
    REQUIRE(result.token_pos[0] == strlen(""));
    // 3. The second token is " pat".
    REQUIRE(model.GetTokenPiece(result.token_id[1]) == " pat");
    REQUIRE(result.token_pos[1] == strlen("I"));
    // 4. The third token is " ubuntu".
    REQUIRE(model.GetTokenPiece(result.token_id[2]) == " ubuntu");
    REQUIRE(result.token_pos[2] == strlen("I pat"));
  }
  SECTION("The model tokenizes Chinese sentence correctly") {
    auto result = tokenizer.Tokenize("像風一样");
    // 1. The result has 4 tokens.
    REQUIRE(result.size == 4);
    REQUIRE(result.size == result.token_id.size());
    REQUIRE(result.size == result.token_pos.size());
    REQUIRE(result.size == result.token_size.size());
    // 2. The first token is "像".
    REQUIRE(model.GetTokenPiece(result.token_id[0]) == "像");
    REQUIRE(result.token_pos[0] == strlen(""));
    // 3. The second token is "風".
    REQUIRE(model.GetTokenPiece(result.token_id[1]) == "風");
    REQUIRE(result.token_pos[1] == strlen("像"));
    // 4. The third token is "一".
    REQUIRE(model.GetTokenPiece(result.token_id[2]) == "一");
    REQUIRE(result.token_pos[2] == strlen("像風"));
    // 5. The fourth token is "样".
    REQUIRE(model.GetTokenPiece(result.token_id[3]) == "样");
    REQUIRE(result.token_pos[3] == strlen("像風一"));
  }
  SECTION("The model tokenizes an empty string correctly") {
    auto result = tokenizer.Tokenize("");
    // The result has 0 token.
    REQUIRE(result.size == 0);
    REQUIRE(result.size == result.token_id.size());
    REQUIRE(result.size == result.token_pos.size());
    REQUIRE(result.size == result.token_size.size());
  }
}

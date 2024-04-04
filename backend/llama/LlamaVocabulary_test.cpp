#include "catch2/catch_test_macros.hpp"

#include "config/Config.h"
#include "llama/LlamaContext.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaVocabulary.h"

TEST_CASE("The vocabulary is the same as llama.cpp's original implementation", "[llama][vocabulary]") {
  auto config = muton::playground::llm::Config::Read("config-test.json");
  auto vocab = muton::playground::llm::LlamaVocabulary::FromGguf(config->getModel().cStr());

  // Construct model using original llama.cpp
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  muton::playground::llm::LlamaContext context(params, model);

  // Compare vocabulary with llama.cpp
  REQUIRE(vocab.GetSize() == llama_n_vocab(model.Get()));
  for (size_t i = 0; i < vocab.GetSize(); ++i) {
    REQUIRE(vocab.GetTokenText(i) == llama_token_get_text(model.Get(), i));
    REQUIRE(vocab.GetTokenScore(i) == llama_token_get_score(model.Get(), i));
    REQUIRE(vocab.GetTokenType(i) == llama_token_get_type(model.Get(), i));
    std::string piece;
    piece.resize(-llama_token_to_piece(model.Get(), i, nullptr, 0));
    llama_token_to_piece(model.Get(), i, piece.data(), piece.size());
    REQUIRE(vocab.GetTokenPiece(i) == piece);
  }

}

TEST_CASE("The vocabulary loads correctly", "[llama][vocabulary]") {
  // Build the model (llama-2-7b.Q4_0.gguf).
  auto config = muton::playground::llm::Config::Read("config-test.json");
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  SECTION("The model gets vocabulary correctly") {
    auto vocab = model.GetVocabulary();
    // Test cases for LlamaModel::GetVocabulary():
    // 1. The vocabulary size is 32000, that is, vocab.size == vocab.strings.size == vocab.scores.size == vocab.pieces.size == 32000.
    REQUIRE(vocab.GetSize() == 32000);
    // 2. The 0 token is <unk> with score 0.
    REQUIRE(vocab.GetTokenText(0) ==  "<unk>");
    REQUIRE(vocab.GetTokenScore(0) == 0);
    // 3. The 29902nd token is I with score -29643.
    REQUIRE(vocab.GetTokenPiece(29902) == "I");
    REQUIRE(vocab.GetTokenText(29902) == "I");
    REQUIRE(vocab.GetTokenScore(29902) == -29643);
    // 4. The 5031st token is pat with score -4772.
    REQUIRE(vocab.GetTokenPiece(5031) == "pat");
    REQUIRE(vocab.GetTokenText(5031) ==  "pat");
    REQUIRE(vocab.GetTokenScore(5031) == -4772);
    // 5. The 8767th token is ubuntu with score -8508.
    REQUIRE(vocab.GetTokenPiece(8767) == "ubuntu");
    REQUIRE(vocab.GetTokenText(8767) == "ubuntu");
    REQUIRE(vocab.GetTokenScore(8767) == -8508);
  }
}

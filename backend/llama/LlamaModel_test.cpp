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

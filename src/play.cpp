#include <fstream>

#include <fmt/format.h>
#include <llama.h>

#include "config/Config.h"
#include "llama/LlamaContext.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"
#include "llama/LlamaTokenizer.h"

int main() {
  auto config = muton::playground::llm::Config::Read();
  auto cfg_model = config->getModel();
  auto cfg_params = config->getParams();
  auto cfg_eval = config->getEval();
  auto cfg_predict = config->getPredict();

  muton::playground::llm::LlamaScope backend(false);
  muton::playground::llm::LlamaParams params{cfg_params};
  muton::playground::llm::LlamaModel model{cfg_model.cStr(), params};
  muton::playground::llm::LlamaContext context{params, model};

  muton::playground::llm::LlamaTokenizer tokenizer{model};
  std::ifstream prompt_file("prompt.txt", std::ios::in);
  std::string prompt{std::istreambuf_iterator<char>(prompt_file), std::istreambuf_iterator<char>()};
  auto prompt_tokenized = tokenizer.Tokenize(prompt);
  for (auto token : prompt_tokenized.token_id) {
    fmt::print("{}", llama_token_to_str_with_model(model.Get(), token));
  }

  context.FeedBos();
  context.Feed(prompt_tokenized.token_id);
  while (true) {
    if (!context.Eval(cfg_eval)) {
      break;
    }
    auto next_token = context.Predict(cfg_predict);
    fmt::print("{}", llama_token_to_str_with_model(model.Get(), next_token));
    static_cast<void>(std::fflush(stdout));
    if (!context.Feed(next_token)) {
      break;
    }
    if (next_token == llama_token_eos()) {
      break;
    }
  }
  fmt::print("\n");
}

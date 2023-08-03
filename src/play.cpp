#include <thread>
#include <vector>

#include <fmt/format.h>
#include <llama.h>

#include "Config.h"
#include "llama/LlamaContext.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"
#include "llama/LlamaTokenizer.h"

int main() {
  auto threads = std::thread::hardware_concurrency();

  auto config = muton::playground::llm::Config::Read();
  fmt::print("model path: {}\n", config->getModel().cStr());
  fmt::print("context length: {}\n", config->getContextLength());

  muton::playground::llm::LlamaScope backend(false);
  muton::playground::llm::LlamaParams params{muton::playground::llm::LlamaParams::Default()};

  params->n_ctx = config->getContextLength();
  params->n_batch = config->getBatchSize();
  params->n_gpu_layers = config->getGpuLayers();

  muton::playground::llm::LlamaModel model{config->getModel().cStr(), params};
  muton::playground::llm::LlamaContext context{model, params};

  muton::playground::llm::LlamaContext::PredictOption predict_option({
      .repeat_penalty_size = 64,
      .repeat_penalty = 1.1F,
      .alpha_presence = 0.0F,
      .alpha_frequency = 0.0F,
      .top_k = 40,
      .tail_free_z = 1.0F,
      .typical_p = 1.0F,
      .top_p = 0.95F,
      .temperature = 0.8F,
  });
  muton::playground::llm::LlamaContext::EvalOption eval_option(
      {.batch_size = 512, .thread_count = std::thread::hardware_concurrency()});

  muton::playground::llm::LlamaTokenizer tokenizer{model};

  std::string prompt{
      " [INST] <<SYS>> "
      "You are a helpful assistant. "
      "<</SYS>> "
      "Hello! What's your name? [/INST] "};
  auto prompt_tokenized = tokenizer.Tokenize(prompt);

  context.FeedBos();
  context.Feed(prompt_tokenized.token_id);
  while (true) {
    if (!context.Eval(eval_option)) {
      break;
    }
    auto next_token = context.Predict(predict_option);
    fmt::print("{}", llama_token_to_str_with_model(model, next_token));
    if (!context.Feed(next_token)) {
      break;
    }
    if (next_token == llama_token_eos()) {
      break;
    }
  }
}

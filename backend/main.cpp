#include "capnp/ez-rpc.h"

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"
#include "service/AppServer.h"

int main() {
  auto config = muton::playground::llm::Config::Read();
  muton::playground::llm::LlamaScope scope(true);
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  capnp::EzRpcServer server(kj::heap<muton::playground::llm::AppServer>(params, model), "127.0.0.1:2333");
  auto& wait_scope = server.getWaitScope();
  kj::NEVER_DONE.wait(wait_scope);
}

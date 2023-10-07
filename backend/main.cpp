#include "capnp/ez-rpc.h"

#include "config/Config.h"
#include "initialize.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "service/AppServer.h"

int main() {
  muton::playground::llm::Initialize({});
  auto config = muton::playground::llm::Config::Read();
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  capnp::EzRpcServer server(kj::heap<muton::playground::llm::AppServer>(params, model), config->getBind().asString());
  auto& wait_scope = server.getWaitScope();
  kj::NEVER_DONE.wait(wait_scope);
}

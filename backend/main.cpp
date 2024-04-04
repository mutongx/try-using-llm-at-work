#include "capnp/ez-rpc.h"
#include "llama.h"
#include "oniguruma.h"

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "service/AppServer.h"

class MainGuard {
 public:
  MainGuard() {
    auto* encoding_ptr = &OnigEncodingUTF8;
    llama_backend_init();
    onig_initialize(&encoding_ptr, 1);
  }
  ~MainGuard() {
    onig_end();
    llama_backend_free();
  }
  MainGuard(MainGuard const&) = delete;
  MainGuard(MainGuard&&) = delete;
  MainGuard& operator=(MainGuard const&) = delete;
  MainGuard& operator=(MainGuard&&) = delete;
};

int main() {
  MainGuard guard;
  auto config = muton::playground::llm::Config::Read();
  muton::playground::llm::LlamaParams params(config->getParams());
  muton::playground::llm::LlamaModel model(config->getModel().cStr(), params);
  capnp::EzRpcServer server(kj::heap<muton::playground::llm::AppServer>(params, model), config->getBind().asString());
  auto& wait_scope = server.getWaitScope();
  kj::NEVER_DONE.wait(wait_scope);
}

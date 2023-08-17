#include "AppServer.h"
#include "ModelServer.h"

#include "capnp/rpc.h"

namespace muton::playground::llm {

AppServer::AppServer(LlamaParams& params, LlamaModel& model) : params_(params), model_(model) {}

kj::Promise<void> AppServer::getModel(AppServer::GetModelContext context) {
  context.getResults().setModel(kj::heap<ModelServer>(params_, model_));
  return kj::READY_NOW;
}

}  // namespace muton::playground::llm

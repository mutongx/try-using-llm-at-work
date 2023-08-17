#ifndef MUTON_PLAYGROUND_LLM_SERVICE_APP_SERVER_H
#define MUTON_PLAYGROUND_LLM_SERVICE_APP_SERVER_H

#include "config/Config.h"
#include "llama/LlamaModel.h"
#include "llama/LlamaParams.h"
#include "llama/LlamaScope.h"

#include "proto/service.capnp.h"

namespace muton::playground::llm {

class AppServer : public proto::App::Server {
 public:
  AppServer(LlamaParams& params, LlamaModel& model);
  AppServer(AppServer&&) = delete;
  AppServer(AppServer const&) = delete;
  AppServer& operator=(AppServer&&) = delete;
  AppServer& operator=(AppServer const&) = delete;
  ~AppServer() = default;

  kj::Promise<void> getModel(GetModelContext context) override;

 private:
  LlamaParams& params_;
  LlamaModel& model_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_SERVICE_APP_SERVER_H

#include "ContextServer.h"

namespace muton::playground::llm {

ContextServer::ContextServer(LlamaParams& params, LlamaModel& model) : context_(params, model) {}

}  // namespace muton::playground::llm

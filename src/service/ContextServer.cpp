#include "ContextServer.h"

namespace muton::playground::llm {

ContextServer::ContextServer(LlamaParams& params, LlamaModel& model) : context_(model, params) {}

}  // namespace muton::playground::llm

#include "LlamaParams.h"

namespace muton::playground::llm {

LlamaParams::LlamaParams(proto::LlamaParams::Reader params) : params_(llama_context_default_params()) {
  params_.n_ctx = static_cast<int>(params.getContextLength());
  params_.n_batch = static_cast<int>(params.getBatchSize());
  params_.n_gpu_layers = static_cast<int>(params.getGpuLayers());
  params_.n_gqa = static_cast<int>(params.getGroupedQueryAttention());
}

llama_context_params LlamaParams::Get() const {
  return params_;
}

llama_context_params const* LlamaParams::operator->() const {
  return &params_;
}

llama_context_params* LlamaParams::operator->() {
  return &params_;
}

}  // namespace muton::playground::llm
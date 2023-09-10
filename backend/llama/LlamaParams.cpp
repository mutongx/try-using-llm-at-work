#include "LlamaParams.h"

namespace muton::playground::llm {

LlamaParams::LlamaParams(proto::LlamaParams::Reader params) : params_(llama_context_default_params()) {
  if (params.getContextLength() != 0) {
    params_.n_ctx = static_cast<int>(params.getContextLength());
  }
  if (params.getBatchSize() != 0) {
    params_.n_batch = static_cast<int>(params.getBatchSize());
  }
  if (params.getGpuLayers() != 0) {
    params_.n_gpu_layers = static_cast<int>(params.getGpuLayers());
  }
  if (params.getRopeFreqBase() != 0.0) {
    params_.rope_freq_base = params.getRopeFreqBase();
  }
  if (params.getRopeFreqScale() != 0.0) {
    params_.rope_freq_scale = params.getRopeFreqScale();
  }
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
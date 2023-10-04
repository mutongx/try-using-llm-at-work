#include "LlamaParams.h"

namespace muton::playground::llm {

LlamaParams::LlamaParams(proto::LlamaParams::Reader params)
    : model_params_(llama_model_default_params()), context_params_(llama_context_default_params()) {
  if (params.getContextLength() != 0) {
    context_params_.n_ctx = static_cast<int>(params.getContextLength());
  }
  if (params.getBatchSize() != 0) {
    context_params_.n_batch = static_cast<int>(params.getBatchSize());
  }
  if (params.getGpuLayers() != 0) {
    model_params_.n_gpu_layers = static_cast<int>(params.getGpuLayers());
  }
  if (params.getRopeFreqBase() != 0.0) {
    context_params_.rope_freq_base = params.getRopeFreqBase();
  }
  if (params.getRopeFreqScale() != 0.0) {
    context_params_.rope_freq_scale = params.getRopeFreqScale();
  }
}

llama_model_params const& LlamaParams::GetModelParams() const {
  return model_params_;
}

llama_context_params const& LlamaParams::GetContextParams() const {
  return context_params_;
}

}  // namespace muton::playground::llm
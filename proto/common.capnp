# SHA256('try-using-llm-at-work:common.capnp')[:16]
@0xc7907fbc29bab4e6;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

struct LlamaParams {
  contextLength @0 :UInt32;
  batchSize @1 :UInt32;
  gpuLayers @2 :UInt32;
}

struct EvalOption {
  batchSize @0 :UInt32;
  threadCount @1 :UInt32;
}

struct PredictOption {
  repeatPenaltySize @0 :UInt32;
  repeatPenalty @1 :Float32;
  alphaPresence @2 :Float32;
  alphaFrequency @3 :Float32;
  topK @4 :UInt32;
  tailFreeZ @5 :Float32;
  typicalP @6 :Float32;
  topP @7 :Float32;
  temperature @8 :Float32;
}

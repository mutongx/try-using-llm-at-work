# SHA256('try-using-llm-at-work:config.capnp')[:16]
@0xd73583e4775512ad;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

struct LlamaParams {
  contextLength @0 :UInt32;
  batchSize @1 :UInt32;
  gpuLayers @2 :UInt32;
  groupedQueryAttention @3 :UInt32;
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

struct Config {
  model @0 :Text;
  params @1 :LlamaParams;
  eval @2 :EvalOption;
  predict @3 :PredictOption;
}

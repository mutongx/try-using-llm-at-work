# SHA256('try-using-llm-at-work:proto.capnp')[:16]
@0xcedc66bd11c8ba6d;

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

interface App {
  getModel @0 () -> (model :Model);
}

interface Model {
  newTokenizer @0 () -> (tokenizer :Tokenizer);
  newContext @1 () -> (context :Context);
}

interface Tokenizer {
  struct TokenizeResult {
    size @0 :UInt32;
    tokenId @1 :List(Int32);
    tokenPos @2 :List(UInt32);
    tokenSize @3 :List(UInt32);
  }
  tokenize @0 (text :Text) -> (tokens :TokenizeResult);
}

interface Context {
  feed @0 (tokens :List(Int32)) -> (context :Context);
  feedBos @1 () -> (context :Context);
  feedEos @2 () -> (context :Context);
  eval @3 () -> (context :Context);
  predict @4 (callback :PredictCallback) -> (context :Context);
  predictUntilEos @5 (callback :PredictCallback) -> (context :Context);
  interface PredictCallback {
    callback @0 (token :Int32) -> ();
  }
}

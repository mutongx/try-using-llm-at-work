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

struct Token {
  enum TokenType {
    normal @0;
    beginOfStream @1;
    endOfStream @2;
  }
  id @0 :Int32;
  type @1 :TokenType;
  str @2 :Text;
}

interface App {
  getModel @0 () -> (model :Model);
}

interface Model {
  newTokenizer @0 () -> (tokenizer :Tokenizer);
  newContext @1 () -> (context :Context);
}

interface Tokens {
  getSize @0 () -> (result :UInt32);
  getTokenId @1 () -> (result :List(Int32));
  getTokenPos @2 () -> (result :List(UInt32));
  getTokenSize @3 () -> (result :List(UInt32));
  getInternalPtr @4 () -> (ptr :UInt64);
}

interface Tokenizer {
  tokenize @0 (text :Text) -> (tokens :Tokens);
}

interface Context {
  nop @0 () -> (success :Bool, context :Context);
  feedTokens @1 (tokens :Tokens, begin :Int32 = 0, end :Int32 = -1) -> (success :Bool, context :Context);
  feedToken @2 (token :Int32) -> (success :Bool, context: Context);
  feedBos @3 () -> (success :Bool, context :Context);
  feedEos @4 () -> (success :Bool, context :Context);
  eval @5 (option :EvalOption) -> (success :Bool, context :Context);
  predict @6 (callback :PredictCallback, option :PredictOption) -> (success :Bool, context :Context);
  predictUntilEos @7 (callback :PredictCallback, evalOption :EvalOption, predictOption :PredictOption) -> (success :Bool, context :Context);
  interface PredictCallback {
    callback @0 (token :Token) -> ();
    done @1 ();
  }
}

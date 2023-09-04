# SHA256('try-using-llm-at-work:service.capnp')[:16]
@0xe16ebe1b28ad9012;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

using Common = import "common.capnp";

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
  eval @5 (option :Common.EvalOption) -> (success :Bool, context :Context);
  predict @6 (callback :PredictCallback, option :Common.PredictOption) -> (success :Bool, context :Context);
  predictUntilEos @7 (callback :PredictCallback, evalOption :Common.EvalOption, predictOption :Common.PredictOption) -> (success :Bool, context :Context);
  interface PredictCallback {
    callback @0 (token :Token) -> ();
    done @1 ();
  }
}

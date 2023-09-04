# SHA256('try-using-llm-at-work:config.capnp')[:16]
@0xd73583e4775512ad;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

using Common = import "common.capnp";

struct Config {
  model @0 :Text;
  params @1 :Common.LlamaParams;
  eval @2 :Common.EvalOption;
  predict @3 :Common.PredictOption;
}

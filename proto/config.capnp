# SHA256('try-using-llm-at-work:config.capnp')[:16]
@0xd73583e4775512ad;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

using Common = import "common.capnp";

struct JenkinsConfig {
  url @0 :Text;
  login @1 :Text;
  password @2 :Text;
}

struct GitHubConfig {
  url @0 :Text;
  login @1 :Text;
  password @2 :Text;
}

struct Config {
  model @0 :Text;
  params @1 :Common.LlamaParams;
  eval @2 :Common.EvalOption;
  predict @3 :Common.PredictOption;
  bind @4 :Text;
  connect @5 :Text;
  jenkins @6 :JenkinsConfig;
  github @7 :GitHubConfig;
}

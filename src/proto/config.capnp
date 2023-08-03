# SHA256('try-using-llm-at-work:config.capnp')[:16]
@0xd73583e4775512ad;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("muton::playground::llm::proto");

struct Config {
  model @0 :Text;
  contextLength @1 :Int32;
  batchSize @2 :Int32;
  gpuLayers @3 :Int32;
}

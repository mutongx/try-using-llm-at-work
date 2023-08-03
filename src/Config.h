#ifndef MUTON_PLAYGROUND_LLM_CONFIG_H
#define MUTON_PLAYGROUND_LLM_CONFIG_H

#include "proto/CapnpMessage.h"
#include "proto/config.capnp.h"

namespace muton::playground::llm {

class Config {
 public:
  using ConfigType =CapnpMessage<capnp::MallocMessageBuilder, proto::Config>;
  static ConfigType Read();
};

}

#endif

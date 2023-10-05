#ifndef MUTON_PLAYGROUND_LLM_CONFIG_H
#define MUTON_PLAYGROUND_LLM_CONFIG_H

#include <string>

#include "capnp/CapnpMessage.h"

#include "config.capnp.h"

namespace muton::playground::llm {

class Config {
 public:
  using ConfigType = CapnpMessage<capnp::MallocMessageBuilder, proto::Config>;
  static ConfigType Read(std::string const& config_path = "config.json");
};

}  // namespace muton::playground::llm

#endif

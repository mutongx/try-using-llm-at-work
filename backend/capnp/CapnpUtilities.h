#ifndef MUTON_PLAYGROUND_LLM_PROTO_CAPNP_UTILITIES_H
#define MUTON_PLAYGROUND_LLM_PROTO_CAPNP_UTILITIES_H

#include <cstddef>

#include <capnp/orphan.h>

namespace muton::playground::llm {

template <typename DstType, typename SrcContainerType>
auto CreateCapnpList(capnp::Orphanage orphanage, SrcContainerType const& src) {
  auto result = orphanage.newOrphan<capnp::List<DstType>>(src.size());
  for (size_t idx = 0; idx < src.size(); ++idx) {
    result.get().set(idx, src[idx]);
  }
  return result;
}

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_PROTO_CAPNP_UTILITIES_H
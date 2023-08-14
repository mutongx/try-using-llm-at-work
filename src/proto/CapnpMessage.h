#ifndef MUTON_PLAYGROUND_LLM_PROTO_CAPNP_MESSAGE_H
#define MUTON_PLAYGROUND_LLM_PROTO_CAPNP_MESSAGE_H

#include <type_traits>
#include <utility>

#include <capnp/message.h>

namespace muton::playground::llm {

template <typename StorageType, typename MessageType>
class CapnpMessage {
 public:
  using MessageRootType = std::conditional_t<std::is_base_of_v<capnp::MessageBuilder, StorageType>,
                                             typename MessageType::Builder,
                                             typename MessageType::Reader>;

  template <typename... Args, typename = std::enable_if_t<std::is_constructible_v<StorageType, Args...>>>
  CapnpMessage(Args&&... args)
      : storage_(std::forward<Args>(args)...), message_(storage_.template getRoot<MessageType>()) {}
  CapnpMessage(CapnpMessage const&);
  CapnpMessage& operator=(CapnpMessage const&);
  CapnpMessage(CapnpMessage&&) = default;
  CapnpMessage& operator=(CapnpMessage&&) = default;
  ~CapnpMessage() = default;
  MessageRootType Root() {
    return message_;
  }
  MessageRootType* operator->() {
    return &message_;
  }

 private:
  StorageType storage_;
  MessageRootType message_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_PROTO_CAPNP_MESSAGE_H
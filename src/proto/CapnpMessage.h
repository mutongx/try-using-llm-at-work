#include <capnp/message.h>

namespace muton::playground {

template <typename BuilderType, typename MessageType>
class CapnpMessage {
 public:
  using MessageBuilderType = MessageType::Builder;

  CapnpMessage() : message_(builder_.template getRoot<MessageType>()) {}
  CapnpMessage(CapnpMessage const&);
  CapnpMessage& operator=(CapnpMessage const&);
  CapnpMessage(CapnpMessage&&) = default;
  CapnpMessage& operator=(CapnpMessage&&) = default;
  MessageBuilderType Root() {
    return message_;
  }
  MessageBuilderType* operator->() {
    return &message_;
  }

 private:
  BuilderType builder_;
  MessageBuilderType message_;
};

}  // namespace muton::playground
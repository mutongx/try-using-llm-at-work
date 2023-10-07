#include "initialize.h"

namespace muton::playground::llm {

class Initializer {
 public:
  Initializer(InitializeOptions options) {
    llama_backend_init(options.llama_numa);
    onig_initialize(options.onig_encodings.data(), static_cast<int>(options.onig_encodings.size()));
  }
  ~Initializer() {
    llama_backend_free();
    onig_end();
  }
  Initializer(Initializer const&) = delete;
  Initializer(Initializer&&) = delete;
  Initializer& operator=(Initializer const&) = delete;
  Initializer& operator=(Initializer&&) = delete;
};

void Initialize(InitializeOptions options) {
  static Initializer i{std::move(options)};
}

}  // namespace muton::playground::llm
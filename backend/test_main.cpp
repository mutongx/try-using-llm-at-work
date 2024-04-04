#include "catch2/catch_session.hpp"
#include "llama.h"
#include "oniguruma.h"

class TestMainGuard {
 public:
  TestMainGuard() {
    auto* encoding_ptr = &OnigEncodingUTF8;
    llama_backend_init();
    onig_initialize(&encoding_ptr, 1);
  }
  ~TestMainGuard() {
    onig_end();
    llama_backend_free();
  }
  TestMainGuard(TestMainGuard const&) = delete;
  TestMainGuard(TestMainGuard&&) = delete;
  TestMainGuard& operator=(TestMainGuard const&) = delete;
  TestMainGuard& operator=(TestMainGuard&&) = delete;
};

int main(int argc, char* argv[]) {
  TestMainGuard guard;
  return Catch::Session().run(argc, argv);
}
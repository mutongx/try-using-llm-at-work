#include "initialize.h"

#include "catch2/catch_session.hpp"

int main(int argc, char* argv[]) {
  muton::playground::llm::Initialize({});
  return Catch::Session().run(argc, argv);
}
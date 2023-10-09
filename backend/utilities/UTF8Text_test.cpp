#include "catch2/catch_test_macros.hpp"

#include "utilities/UTF8Text.h"

TEST_CASE("UTF8Text can parse unicode text", "[unicode]") {
  using UnicodeText = muton::playground::llm::UTF8Text;
  SECTION("Simple ASCII text") {
    std::vector<std::string> expected_str{"T", "e", "s", "t"};
    std::vector<uint32_t> expected_cp{'T', 'e', 's', 't'};
    size_t i = 0;
    for (auto symbol : UnicodeText("Test")) {
      REQUIRE(symbol.str == expected_str[i]);
      REQUIRE(symbol.cp == expected_cp[i]);
      ++i;
    }
    REQUIRE(i == expected_str.size());
  }
  SECTION("Chinese text") {
    std::vector<std::string> expected_str{"你", "好"};
    std::vector<uint32_t> expected_cp{0x4F60, 0x597D};
    size_t i = 0;
    for (auto symbol : UnicodeText("你好")) {
      REQUIRE(symbol.str == expected_str[i]);
      REQUIRE(symbol.cp == expected_cp[i]);
      ++i;
    }
    REQUIRE(i == expected_str.size());
  }
  SECTION("Emoji text") {
    std::vector<std::string> expected_str{"🥰", "😘", "🤣"};
    std::vector<uint32_t> expected_cp{0x1F970, 0x1F618, 0x1F923};
    size_t i = 0;
    for (auto symbol : UnicodeText("🥰😘🤣")) {
      REQUIRE(symbol.str == expected_str[i]);
      REQUIRE(symbol.cp == expected_cp[i]);
      ++i;
    }
    REQUIRE(i == expected_str.size());
  }
  SECTION("Invalid sequence") {}
}

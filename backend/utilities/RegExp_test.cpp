#include "catch2/catch_test_macros.hpp"

#include <iostream>
#include <vector>

#include "utilities/RegExp.h"

TEST_CASE("RegExp::Match works correctly", "[regexp]") {
  using RegExp = muton::playground::llm::RegExp;
  std::string_view pattern{R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};
  std::string_view text{"你好，吃了？   🤯 🍦  Hello  World! It's OK?  "};
  std::vector<std::string> expected{
      "你好", "，", "吃了", "？", "  ", " 🤯", " 🍦", " ", " Hello", " ", " World", "!", " It", "'s", " OK", "?", "  "};
  size_t i = 0;
  for (RegExp re(pattern); auto match : re.Match(text)) {
    REQUIRE(expected[i] == match);
    ++i;
  }
  REQUIRE(i == expected.size());
}

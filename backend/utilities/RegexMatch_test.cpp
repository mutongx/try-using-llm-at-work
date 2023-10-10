#include "catch2/catch_test_macros.hpp"

#include <iostream>
#include <vector>

#include "utilities/RegexMatch.h"

TEST_CASE("RegexMatch works correctly", "[regex]") {
  using RegexMatch = muton::playground::llm::RegexMatch;
  std::string_view pattern{R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"};
  std::string_view text{"你好，吃了？   🤯 🍦  Hello  World! It's OK?  "};
  std::vector<std::string> expected{
      "你好", "，", "吃了", "？", "  ", " 🤯", " 🍦", " ", " Hello", " ", " World", "!", " It", "'s", " OK", "?", "  "};
  size_t i = 0;
  for (auto match : RegexMatch(text, pattern)) {
    REQUIRE(expected[i] == match);
    ++i;
  }
  REQUIRE(i == expected.size());
}

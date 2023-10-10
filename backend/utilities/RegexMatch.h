#ifndef MUTON_PLAYGROUND_LLM_UTILITIES_REGEX_MATCH_H
#define MUTON_PLAYGROUND_LLM_UTILITIES_REGEX_MATCH_H

#include <string_view>

#include "oniguruma.h"

namespace muton::playground::llm {

class RegexMatch {
 public:
  class Iterator {
   public:
    Iterator(std::string_view text, size_t pos, OnigRegex regex, OnigRegion* region);
    bool operator!=(Iterator const& rhs) const;
    Iterator& operator++();
    std::string_view operator*() const;

   private:
    void Update();
    std::string_view text_;
    size_t pos_{};
    size_t size_{};
    OnigRegex regex_;
    OnigRegion* region_;
  };
  RegexMatch(std::string_view text, std::string_view pattern);
  RegexMatch(RegexMatch const&) = delete;
  RegexMatch(RegexMatch&&) = delete;
  RegexMatch& operator=(RegexMatch const&) = delete;
  RegexMatch& operator=(RegexMatch&&) = delete;
  ~RegexMatch();
  [[nodiscard]] Iterator begin() const;
  [[nodiscard]] Iterator end() const;

 private:
  std::string_view text_;
  OnigRegex regex_{};
  OnigRegion* region_{};
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_UTILITIES_REGEX_MATCH_H

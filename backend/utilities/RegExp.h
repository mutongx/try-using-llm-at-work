#ifndef MUTON_PLAYGROUND_LLM_UTILITIES_REG_EXP_H
#define MUTON_PLAYGROUND_LLM_UTILITIES_REG_EXP_H

#include <string_view>

#include "oniguruma.h"

namespace muton::playground::llm {

class RegExp {
 public:
  class Matcher {
   public:
    class Iterator {
     public:
      Iterator(std::string_view text, size_t pos, OnigRegex regex, OnigRegion* region);
      bool operator!=(Iterator const& rhs) const;
      Iterator& operator++();
      std::string_view operator*() const;

     private:
      void Update();
      std::string_view text_{};
      size_t pos_{};
      size_t size_{};
      OnigRegex regex_;
      OnigRegion* region_;
    };
    Matcher(std::string_view text, OnigRegex regex);
    Matcher(Matcher const&) = delete;
    Matcher(Matcher&&) = delete;
    Matcher& operator=(Matcher const&) = delete;
    Matcher& operator=(Matcher&&) = delete;
    ~Matcher();
    [[nodiscard]] Iterator begin() const;
    [[nodiscard]] Iterator end() const;

   private:
    std::string_view text_;
    OnigRegex regex_;
    OnigRegion* region_;
  };
  RegExp(std::string_view pattern);
  RegExp(RegExp const&) = delete;
  RegExp(RegExp&&) = delete;
  RegExp& operator=(RegExp const&) = delete;
  RegExp& operator=(RegExp&&) = delete;
  ~RegExp();

  Matcher Match(std::string_view text);

 private:
  OnigRegex regex_{};
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_UTILITIES_REG_EXP_H

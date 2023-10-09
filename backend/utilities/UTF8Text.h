#ifndef MUTON_PLAYGROUND_LLM_UTILITIES_UNICODE_H
#define MUTON_PLAYGROUND_LLM_UTILITIES_UNICODE_H

#include <string_view>

namespace muton::playground::llm {

class UTF8Text {
 public:
  struct Symbol {
    std::string_view str;
    uint32_t cp;
  };

  class Iterator {
   public:
    Iterator(std::string_view text, size_t pos);
    bool operator!=(Iterator const& rhs) const;
    Iterator& operator++();
    Symbol operator*() const;

   private:
    void Update();

    std::string_view text_;
    size_t pos_;
    size_t symbol_size_{};
    uint32_t code_point_{};
  };
  UTF8Text(std::string_view text);
  [[nodiscard]] Iterator begin() const;
  [[nodiscard]] Iterator end() const;

 private:
  std::string_view text_;
};

}  // namespace muton::playground::llm

#endif  // MUTON_PLAYGROUND_LLM_UTILITIES_UNICODE_H

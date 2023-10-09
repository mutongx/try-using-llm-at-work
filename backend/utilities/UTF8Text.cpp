#include "UTF8Text.h"

#include <stdexcept>

namespace muton::playground::llm {

UTF8Text::UTF8Text(std::string_view text) : text_(text) {}

UTF8Text::Iterator UTF8Text::begin() const {
  return {text_, 0};
}

UTF8Text::Iterator UTF8Text::end() const {
  return {text_, text_.size()};
}

UTF8Text::Iterator::Iterator(std::string_view text, size_t pos) : text_(text), pos_(pos) {
  Update();
}

bool UTF8Text::Iterator::operator!=(UTF8Text::Iterator const& rhs) const {
  return pos_ != rhs.pos_;
}

UTF8Text::Iterator& UTF8Text::Iterator::operator++() {
  pos_ += symbol_size_;
  Update();
  return *this;
}

UTF8Text::Symbol UTF8Text::Iterator::operator*() const {
  return {
      {text_.data() + pos_, text_.data() + pos_ + symbol_size_},
      code_point_,
  };
}

void UTF8Text::Iterator::Update() {
  if (pos_ == text_.size()) {
    return;
  }
  uint8_t first_bit_mask;
  size_t trailing_bytes;
  auto first = static_cast<uint8_t>(text_[pos_]);
  if ((first & 0b11110000U) == 0b11110000U) {
    if (0b11110000U <= first && first <= 0b11110111U) {
      first_bit_mask = 0b00000111U;
      trailing_bytes = 3;
    } else {
      throw std::runtime_error("invalid byte encoding");
    }
  } else if ((first & 0b11100000U) == 0b11100000U) {
    if (0b11100000U <= first && first <= 0b11101111U) {
      first_bit_mask = 0b00001111U;
      trailing_bytes = 2;
    } else {
      throw std::runtime_error("invalid byte encoding");
    }
  } else if ((first & 0b11000000U) == 0b11000000U) {
    if (0b11000000U <= first && first <= 0b11011111U) {
      first_bit_mask = 0b00011111U;
      trailing_bytes = 1;
    } else {
      throw std::runtime_error("invalid byte encoding");
    }
  } else {
    if (0b00000000 <= first && first <= 0b01111111) {
      first_bit_mask = 0b01111111;
      trailing_bytes = 0;
    } else {
      throw std::runtime_error("invalid byte encoding");
    }
  }
  symbol_size_ = trailing_bytes + 1;
  if (pos_ + symbol_size_ > text_.size()) {
    throw std::runtime_error("invalid byte encoding");
  }
  code_point_ = first & first_bit_mask;
  for (size_t i = 0; i < trailing_bytes; ++i) {
    auto byte = static_cast<uint8_t>(text_[pos_ + i + 1]);
    if ((byte >> 6U) != 0b00000010U) {
      throw std::runtime_error("invalid byte encoding");
    }
    code_point_ = code_point_ << 6U;
    code_point_ = code_point_ + (byte & 0b00111111U);
  }
}

}  // namespace muton::playground::llm
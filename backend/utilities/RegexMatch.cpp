#include "RegexMatch.h"

#include <array>
#include <stdexcept>

namespace muton::playground::llm {

RegexMatch::RegexMatch(std::string_view text, std::string_view pattern) : text_(text) {
  if (onig_new(&regex_,
               reinterpret_cast<OnigUChar const *>(pattern.data()),
               reinterpret_cast<OnigUChar const *>(pattern.data() + pattern.size()),
               ONIG_OPTION_NONE,
               ONIG_ENCODING_UTF8,
               ONIG_SYNTAX_DEFAULT,
               nullptr) != ONIG_NORMAL) {
    throw std::runtime_error("onig_new error");
  }
  region_ = onig_region_new();
}

RegexMatch::~RegexMatch() {
  onig_region_free(region_, 1);
  onig_free(regex_);
};

RegexMatch::Iterator RegexMatch::begin() const {
  return {text_, 0, regex_, region_};
}

RegexMatch::Iterator RegexMatch::end() const {
  return {text_, text_.size(), regex_, nullptr};
}

RegexMatch::Iterator::Iterator(std::string_view text, size_t pos, OnigRegex regex, OnigRegion *region)
    : text_(text), pos_(pos), regex_(regex), region_(region) {
  Update();
}

bool RegexMatch::Iterator::operator!=(Iterator const &rhs) const {
  return pos_ != rhs.pos_;
}

RegexMatch::Iterator &RegexMatch::Iterator::operator++() {
  pos_ += size_;
  Update();
  return *this;
}

std::string_view RegexMatch::Iterator::operator*() const {
  return {text_.data() + pos_, size_};
}

void RegexMatch::Iterator::Update() {
  auto const *text_uchar = reinterpret_cast<OnigUChar const *>(text_.data());
  auto search = onig_search(regex_,
                            text_uchar,
                            text_uchar + text_.size(),
                            text_uchar + pos_,
                            text_uchar + text_.size(),
                            region_,
                            ONIG_OPTION_NONE);
  if (search == ONIG_MISMATCH) {
    pos_ = text_.size();
    size_ = 0;
  } else {
    size_ = region_->end[0] - region_->beg[0];
  }
}

}  // namespace muton::playground::llm
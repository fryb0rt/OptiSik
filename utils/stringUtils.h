#pragma once

#include <string>
#include <vector>

/// Splits a string by the given character delimiter
std::vector<std::string> split(const std::string &s, char c,
                               bool keepEmpty = false) {
  std::vector<std::string> result;
  int start = 0, end = int(s.size());
  for (int i = 0; i < end; ++i) {
    if (s[i] == c) {
      if (i - start > 0 || keepEmpty) {
        result.push_back(s.substr(start, i - start));
      }
      start = i + 1;
    }
  }
  if (end - start > 0 || keepEmpty) {
    result.push_back(s.substr(start, end - start));
  }
  return result;
}

/// Parses a list of values of type T from a string, separated by the given
/// character delimiter
template <typename T> std::vector<T> parseList(const std::string &s, char sep) {
  int start = 0, end = int(s.size());
  while (start < end && !std::isdigit(s[start])) {
    ++start;
  }
  while (end > start && !std::isdigit(s[end - 1])) {
    --end;
  }
  std::vector<std::string> spl = split(s.substr(start, end - start), sep);
  std::vector<T> res(spl.size());
  for (int i = 0; i < res.size(); ++i) {
    if constexpr (sizeof(T) == sizeof(long)) {
      res[i] = std::stoi(spl[i]);
    } else {
      res[i] = std::stoll(spl[i]);
    }
  }
  return res;
}
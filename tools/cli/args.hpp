// Minimal command line parsing: "--key value" options, boolean "--flag" switches,
// and positional arguments.
#pragma once

#include <cstdlib>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace cli {

struct Args {
  std::map<std::string, std::string> opts;
  std::vector<std::string> pos;
  std::string error; // set when parsing fails

  bool has(const std::string& k) const { return opts.count(k) != 0; }
  std::string get(const std::string& k, const std::string& def) const {
    auto it = opts.find(k);
    return it == opts.end() ? def : it->second;
  }
  int get_int(const std::string& k, int def) const {
    return has(k) ? std::atoi(opts.at(k).c_str()) : def;
  }
  double get_double(const std::string& k, double def) const {
    return has(k) ? std::atof(opts.at(k).c_str()) : def;
  }
};

// Parse argv[first..]. Names in `flags` take no value.
inline Args parse_args(int argc, char** argv, int first, const std::set<std::string>& flags) {
  Args a;
  for (int i = first; i < argc; ++i) {
    const std::string s = argv[i];
    if (s.rfind("--", 0) == 0) {
      const std::string key = s.substr(2);
      if (flags.count(key)) {
        a.opts[key] = "1";
      } else if (i + 1 < argc) {
        a.opts[key] = argv[++i];
      } else {
        a.error = "missing value for --" + key;
        return a;
      }
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

// "224x224" -> (224, 224). Returns false on a malformed string.
inline bool parse_size(const std::string& s, int* w, int* h) {
  const size_t x = s.find('x');
  if (x == std::string::npos)
    return false;
  *w = std::atoi(s.substr(0, x).c_str());
  *h = std::atoi(s.substr(x + 1).c_str());
  return *w > 0 && *h > 0;
}

// "1,16,64" -> {1, 16, 64}
inline std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty())
      out.push_back(std::atoi(item.c_str()));
  return out;
}

} // namespace cli

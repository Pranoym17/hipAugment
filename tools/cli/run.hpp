#pragma once

#include "args.hpp"

namespace cli {

// `hipaug run`: apply one op to every image in a directory. Returns the process exit code.
int cmd_run(const Args& args);

} // namespace cli

#pragma once

#include "args.hpp"

namespace bench {

// `hipaug bench`: time every op on synthetic batches and write a CSV.
// Returns the process exit code.
int cmd_bench(const cli::Args& args);

} // namespace bench

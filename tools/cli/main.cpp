// hipaug: command line front end.
//
//   hipaug run --op <name> --batch <N> --size <WxH> <input_dir> <output_dir>
//   hipaug bench
#include <cstdio>
#include <cstring>

#include "args.hpp"
#include "bench.hpp"
#include "run.hpp"

namespace {

void usage() {
  std::puts(
      "hipaug - batched GPU image preprocessing\n"
      "\n"
      "  hipaug run --op <name> --batch <N> --size <WxH> [options] <input_dir> <output_dir>\n"
      "      --op       resize | gaussian_blur | crop_mirror_normalize | fused_rcmn\n"
      "      --batch    images per batch (default 16)\n"
      "      --size     output size, for example 224x224 (not used by gaussian_blur)\n"
      "      --backend  hip | cpu (default hip when built with HIP)\n"
      "      --ksize    blur kernel size, 3 or 5 (default 5)\n"
      "      --sigma    blur sigma (default 1.0)\n"
      "      --mirror   flip every image in x (crop_mirror_normalize, fused_rcmn)\n"
      "    Reads every .jpg, .jpeg, .png and .bmp in input_dir and writes PNGs to output_dir.\n"
      "    Float outputs are mapped back to 0..255 before they are saved.\n"
      "\n"
      "  hipaug bench [options]\n"
      "      --out        CSV file to write (default results.csv)\n"
      "      --batches    comma separated batch sizes (default 1,16,64,256)\n"
      "      --size       output size (default 224x224)\n"
      "      --src-min    smallest source image side (default 256)\n"
      "      --src-max    largest source image side (default 512)\n"
      "      --warmup     GPU warmup runs (default 10)\n"
      "      --runs       GPU timed runs (default 100)\n"
      "      --cpu-runs   CPU timed runs (default 10)\n"
      "      --peak-gbps  peak memory bandwidth used for the share column (default 5300)\n"
      "      --no-cpu     skip the CPU reference\n"
      "      --no-gpu     skip the HIP kernels\n");
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) {
    usage();
    return argc < 2 ? 2 : 0;
  }
  const std::string cmd = argv[1];
  const cli::Args args = cli::parse_args(argc, argv, 2, {"mirror", "no-cpu", "no-gpu"});
  if (!args.error.empty()) {
    std::fprintf(stderr, "hipaug: %s\n", args.error.c_str());
    return 2;
  }
  if (cmd == "run")
    return cli::cmd_run(args);
  if (cmd == "bench")
    return bench::cmd_bench(args);
  std::fprintf(stderr, "hipaug: unknown command '%s'\n\n", cmd.c_str());
  usage();
  return 2;
}

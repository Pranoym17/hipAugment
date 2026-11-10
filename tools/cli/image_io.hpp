// Image file loading and saving on the CPU, through stb_image. Decoding is outside
// the library's scope; the CLI does it so the ops have something to work on.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cli {

struct Image {
  int w = 0, h = 0;
  std::vector<uint8_t> rgb; // interleaved, w * h * 3
};

// Load any format stb_image supports as 8-bit RGB. Returns false on failure.
bool load_image(const std::string& path, Image* out);

// Save 8-bit RGB as PNG. `row_stride` is the distance in bytes between rows.
bool save_png(const std::string& path, int w, int h, const uint8_t* rgb, int row_stride);

} // namespace cli

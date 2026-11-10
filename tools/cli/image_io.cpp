#include "image_io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

namespace cli {

bool load_image(const std::string& path, Image* out) {
  int w = 0, h = 0, comp = 0;
  unsigned char* px = stbi_load(path.c_str(), &w, &h, &comp, 3);
  if (!px)
    return false;
  out->w = w;
  out->h = h;
  out->rgb.assign(px, px + static_cast<size_t>(w) * h * 3);
  stbi_image_free(px);
  return true;
}

bool save_png(const std::string& path, int w, int h, const uint8_t* rgb, int row_stride) {
  return stbi_write_png(path.c_str(), w, h, 3, rgb, row_stride) != 0;
}

} // namespace cli

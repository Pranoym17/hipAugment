// Fiber scheduler for the HIP emulation. See hip/hip_runtime.h.
#include <ucontext.h>

#include <vector>

#include "hip/hip_runtime.h"

dim3 threadIdx, blockIdx, blockDim, gridDim;

namespace hipemu {
namespace {

constexpr size_t kStackBytes = 64 * 1024;

struct Fiber {
  ucontext_t ctx;
  bool done = true;
};

ucontext_t g_scheduler;
Fiber* g_current = nullptr;
const std::function<void()>* g_body = nullptr;

void fiber_entry() {
  (*g_body)();
  g_current->done = true; // returning switches back to the scheduler through uc_link
}

} // namespace

void syncthreads() {
  // Suspend this thread. The scheduler resumes it only after every other thread in
  // the block has also reached a barrier or returned.
  swapcontext(&g_current->ctx, &g_scheduler);
}

void launch(dim3 grid, dim3 block, const std::function<void()>& body) {
  const size_t threads = static_cast<size_t>(block.x) * block.y * block.z;
  std::vector<Fiber> fibers(threads);
  std::vector<char> stacks(threads * kStackBytes);
  g_body = &body;
  gridDim = grid;
  blockDim = block;

  for (uint32_t bz = 0; bz < grid.z; ++bz)
    for (uint32_t by = 0; by < grid.y; ++by)
      for (uint32_t bx = 0; bx < grid.x; ++bx) {
        blockIdx = dim3(bx, by, bz);
        for (size_t t = 0; t < threads; ++t) {
          Fiber& f = fibers[t];
          getcontext(&f.ctx);
          f.ctx.uc_stack.ss_sp = &stacks[t * kStackBytes];
          f.ctx.uc_stack.ss_size = kStackBytes;
          f.ctx.uc_link = &g_scheduler;
          makecontext(&f.ctx, fiber_entry, 0);
          f.done = false;
        }
        // Each pass runs every live thread up to its next barrier (or to the end).
        bool any_live = true;
        while (any_live) {
          any_live = false;
          for (size_t t = 0; t < threads; ++t) {
            Fiber& f = fibers[t];
            if (f.done)
              continue;
            threadIdx = dim3(static_cast<uint32_t>(t % block.x),
                             static_cast<uint32_t>((t / block.x) % block.y),
                             static_cast<uint32_t>(t / (static_cast<size_t>(block.x) * block.y)));
            g_current = &f;
            swapcontext(&g_scheduler, &f.ctx);
            any_live = any_live || !f.done;
          }
        }
      }
  g_body = nullptr;
  g_current = nullptr;
}

} // namespace hipemu

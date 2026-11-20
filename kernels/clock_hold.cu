// One short piece of card work that keeps the card's clocks up between a split's forwards.
//
// WHY THIS FILE EXISTS
// --------------------
// A split's segment works ~26 ms and then waits ~400 ms for the far end; across that wait the card's
// own governor drops its clocks (an RTX 3070 Laptop read 210-420 MHz through a Qwen3.8-27B split's
// replies, against 1755 at full boost), so the next forward ran up to 7 times slower: 26 ms right after
// a slow one, then 49, 58, 65, 77, 88, 180 ms as the clocks fell (2026-11-20, split27_prof). Locking
// the clocks (`nvidia-smi -lgc`) needs administrator rights a node does not have; this is the user-space
// answer — work too small to matter, often enough that the governor reads the card as busy
// (`inference::clock_hold`). A probe on that card held 1755 MHz with 2 ms pieces at half the time.
//
// ONE block of one warp, so the card's other units stay free and the power it draws stays small. It
// ends at `ns` of the card's own timer, or after `max_cycles` of the processor's clock — whichever
// comes first — so a timer that stood still could never hold the card's stream near Windows'
// two-second watchdog (TDR).
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>

extern "C" __global__ void clock_hold_piece(const unsigned long long ns,
                                            const unsigned long long max_cycles) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    const long long c0 = clock64();
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns && (unsigned long long)(clock64() - c0) < max_cycles);
}

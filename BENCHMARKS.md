# oceanlib — Benchmark results

Only measured numbers go in this file. Nothing here is estimated or projected.

## Test machine

| Field       | Value                                                              |
|-------------|--------------------------------------------------------------------|
| CPU         | Intel Core i7-14700HX (8 P-cores + 12 E-cores, 28 threads)         |
| ISA         | SSE2 / AVX2 available; **no** AVX-512 (fused off on Raptor Lake)   |
| OS          | Windows 11 Home 26200                                              |
| Compiler    | MSVC 19.44.35208 (VS 2022 17.14), x64                              |
| Flags       | `/O2 /Ob2 /DNDEBUG -std:c++20 -MD /W4 /permissive- /fp:precise`    |
| Build type  | CMake `Release`, Ninja generator                                   |

Laptop CPU: results are subject to thermal and power-state variation. The
benchmark reports the **median** of many iterations, because a scheduler hiccup
or a thermal blip produces occasional large outliers that drag a mean around
and say nothing about the code. The minimum is reported alongside as the best
case the machine can reach.

Run it with `build/bench/ocean_bench.exe`.

---

## Baseline: single-threaded, scalar

Measured 2026-09-17. Patch 200 m, wind 12 m/s, choppiness 1.0. Times are for
one complete `Ocean::update()`.

| N     | total (ms) | best (ms) | evolve (ms) | FFT (ms) | finalise (ms) | Mcell/s |
|------:|-----------:|----------:|------------:|---------:|--------------:|--------:|
| 64x   |      0.243 |     0.237 |       0.084 |    0.152 |         0.009 |    16.9 |
| 128x  |      1.171 |     1.136 |       0.344 |    0.770 |         0.037 |    14.0 |
| 256x  |      4.998 |     4.838 |       1.304 |    3.461 |         0.153 |    13.1 |
| 512x  |     23.620 |    22.896 |       5.346 |   17.176 |         0.634 |    11.1 |

### What this says

**The FFT dominates: ~73% of frame time at 512 squared.** Four inverse 2D
transforms, each N row transforms plus N column transforms. This is where SIMD
effort belongs first.

**Spectrum evolution is ~23%.** That stage is one `sinf`/`cosf` pair per cell -
262 144 of them at 512 squared - plus the eight spectrum derivations. A
vectorised sine/cosine is the obvious lever, and it would have a second
benefit: a polynomial approximation we write ourselves is bit-identical across
platforms, which would close the last cross-runtime determinism gap noted in
ADR-003.

**Finalisation is ~3%** and is not worth optimising. It is memory-bound,
writing 8 MB per frame at 512 squared, and already close to what the store
bandwidth allows.

**Scaling.** Cell throughput falls from 16.9 to 11.1 Mcell/s as N grows, which
is the expected shape: per-cell work grows as `log N` from the extra FFT
stages, and the working set stops fitting in cache.

---

## Multithreaded (scalar)

Measured 2026-09-17, same settings, `hardware_concurrency = 28`. "serial" is
`thread_count = 1`; "threaded" is the default (one worker per hardware thread).
Stage columns are the **serial** costs, shown to locate the work rather than to
sum to the threaded total.

| N     | serial (ms) | threaded (ms) | speedup | evolve (ms) | FFT (ms) | finalise (ms) | Mcell/s |
|------:|------------:|--------------:|--------:|------------:|---------:|--------------:|--------:|
| 64x   |       0.251 |         0.255 |   0.98x |       0.083 |    0.155 |         0.009 |    16.1 |
| 128x  |       1.199 |         0.527 |   2.28x |       0.368 |    0.791 |         0.037 |    31.1 |
| 256x  |       5.144 |         0.987 |   5.21x |       1.357 |    3.563 |         0.160 |    66.4 |
| 512x  |      25.558 |         2.919 |   8.76x |       5.471 |   19.360 |         0.636 |    89.8 |

### What this says

**512 squared now fits a 60 Hz budget with room to spare: 2.92 ms.** It was
23.6 ms. 256 squared at 0.99 ms leaves essentially the whole frame to the
renderer.

**Speedup is 8.76x, not 28x, and that is expected.** Four reasons, in rough
order of size:

1. *The workload is memory-bound, not compute-bound.* At 512 squared the four
   complex fields are 8 MB and every FFT stage streams all of it. Cores added
   past the point where the memory system saturates buy very little.
2. *This is a hybrid CPU.* Of the 28 hardware threads, only 8 are P-core
   primaries; 12 are E-cores with substantially lower throughput, and 8 more
   are P-core hyperthreads. Counting "28" as 28 units of capacity was never
   right.
3. *Hyperthreading gives well under 2x* on a workload that already keeps the
   load/store units busy.
4. *Four barriers per frame.* Every dispatch ends with all threads waiting for
   the slowest chunk.

**The serial-path threshold.** An earlier build threaded unconditionally and
64 squared got *slower*: 0.243 ms serial against 0.296 ms threaded, because four
condition-variable round trips cost more than the work being distributed. The
library now runs serially below 8192 cells, and does not create a thread pool
it would never dispatch to. 64 squared is now a tie (0.98x is measurement
noise), which is the correct outcome: threading should never make things worse.

**The work did not change.** Serial stage costs are within noise of the
single-threaded baseline above, as they should be - threading moves the wall
clock, not the arithmetic. The FFT is still ~75% of the work, so it remains
the target for SIMD.

---

## SIMD (AVX2) + multithreaded

Measured 2026-09-17, same settings. Detected SIMD level: AVX2.

| N     | serial (ms) | threaded (ms) | speedup | evolve (ms) | FFT (ms) | finalise (ms) | Mcell/s |
|------:|------------:|--------------:|--------:|------------:|---------:|--------------:|--------:|
| 64x   |       0.178 |         0.179 |   0.99x |       0.083 |    0.091 |         0.009 |    22.8 |
| 128x  |       0.726 |         0.386 |   1.88x |       0.352 |    0.345 |         0.037 |    42.5 |
| 256x  |       2.980 |         0.588 |   5.07x |       1.335 |    1.485 |         0.150 |   111.4 |
| 512x  |      13.371 |         1.913 |   6.99x |       5.334 |    7.071 |         0.623 |   137.0 |

### Progress at 512 squared

| build                          | serial (ms) | threaded (ms) |
|--------------------------------|------------:|--------------:|
| scalar, single-threaded        |      23.620 |             - |
| scalar + threads               |      25.558 |         2.919 |
| AVX2 (naive) + threads         |      18.301 |         2.422 |
| AVX2 + batched columns + threads |    13.371 |     **1.913** |

**12.3x overall from where we started.** The FFT stage went 17.176 ms to
7.071 ms serial, a 2.43x improvement.

### Why AVX2 gave 1.62x, not 8x - and what fixed it

The first AVX2 build sped the FFT up by only 1.62x. Rather than accept that,
the row and column passes were timed separately, per field, at 512 squared:

| pass    | scalar | AVX2 (naive) | gain  |
|---------|-------:|-------------:|------:|
| rows    | 1.608  |        0.970 | 1.66x |
| columns | 2.632  |        2.074 | 1.27x |

The butterflies are the same code in both passes, so the difference had to be
memory access. The column pass walks with stride N - 2 KB at 512 squared - so
gathering one column touched a separate cache line for every one of its N
elements, used 4 bytes of each 64-byte line, and could not be vectorised at all.

**The fix: transform 8 adjacent columns at once.** Each element then becomes 8
contiguous floats, so a fetched cache line is actually used. It also removes
the other weakness: batching multiplies the memory stride by 8, so the stride
is at least 8 from the very first stage, and the three early stages that
previously fell back to scalar (s = 1, 2, 4 are below the 8-float vector width)
now vectorise like the rest.

One kernel serves both cases. The only change needed was splitting the
*memory* stride from the *logical* twiddle step, because a twiddle depends on
position within a sequence and not on which lane of the batch it sits in.

Result: FFT 11.974 ms to 7.071 ms serial, a further 1.69x.

### A bug the benchmark caught

Immediately after the batching landed, the stage columns summed to 12.9 ms
while the serial total read 18.2 ms. The 5.3 ms gap was real: the threaded
pipeline's column task still called the single-column `transform()`, so only
the benchmark's own `transform_2d` path was using the batched routine. The
arithmetic not adding up is what exposed it - a reason to print a stage
breakdown rather than one total.

### Where the time goes now

At 512 squared, serial: **evolve 5.33 ms (40%)**, FFT 7.07 ms (53%),
finalise 0.62 ms (5%). Spectrum evolution is now the largest single stage per
unit of remaining opportunity, and it is still entirely scalar - one `sinf` and
one `cosf` per cell, 262 144 of each per frame. A vectorised polynomial
sine/cosine is the next lever, and it would also close the cross-runtime
determinism gap from ADR-003, since a polynomial we evaluate ourselves is
bit-identical everywhere while libm is not.

---

## Vectorised spectrum evolution

Measured 2026-09-17, same machine and settings.

| N     | serial (ms) | threaded (ms) | speedup | evolve (ms) | FFT (ms) | finalise (ms) | Mcell/s |
|------:|------------:|--------------:|--------:|------------:|---------:|--------------:|--------:|
| 64x   |       0.098 |         0.098 |   1.00x |       0.007 |    0.083 |         0.009 |    41.8 |
| 128x  |       0.401 |         0.343 |   1.17x |       0.029 |    0.331 |         0.037 |    47.8 |
| 256x  |       1.664 |         0.503 |   3.31x |       0.119 |    1.359 |         0.222 |   130.2 |
| 512x  |       8.378 |         1.426 |   5.88x |       0.449 |    7.002 |         0.623 |   183.8 |

### The evolve stage, in three steps

The stage was 5.334 ms serial at 512 squared. Each change was measured
separately rather than bundled, because that is the only way to know which one
actually paid:

| change                                   | evolve (ms) | gain  |
|------------------------------------------|------------:|------:|
| baseline (`std::sinf`/`cosf`, `std::fmod`) |       5.334 |     - |
| own polynomial sincos                    |       4.024 | 1.33x |
| `d - floor(d)` instead of `std::fmod`     |       3.439 | 1.17x |
| AVX2 kernel                              |   **0.449** | 7.66x |
| **total**                                |             | **11.9x** |

Two things worth noting:

**The polynomial sincos was not the big win — `fmod` was comparable.** The
expectation going in was that the transcendentals dominated. Replacing them
gained 1.33x; replacing a single `std::fmod` in the same loop gained another
1.17x on top. `fmod` is a libm call that cannot be inlined into a vector loop,
so it was both slow and an outright blocker to vectorising the stage at all.

**AVX2 gave 7.66x, close to the theoretical 8x** - in sharp contrast to the
FFT, which gave only 1.62x before the memory layout was fixed. The difference
is the access pattern: evolve streams eight input arrays and eight output
arrays with unit stride, so every cache line fetched is fully used. The FFT's
column pass was touching one float per line. Same instruction set, same vector
width, wildly different results - which is the whole argument for profiling
rather than assuming that wider registers mean proportionally faster code.

Caveat on this number: the benchmark loops over a 16 MB working set, which fits
in this CPU's 33 MB L3. A real frame shares L3 with a renderer, so the evolve
stage would be somewhat slower in situ.

## Overall progress at 512 squared

| build                                         | serial (ms) | threaded (ms) |
|-----------------------------------------------|------------:|--------------:|
| scalar, single-threaded                       |      23.620 |             - |
| + threads                                     |      25.558 |         2.919 |
| + AVX2 FFT (naive)                            |      18.301 |         2.422 |
| + batched FFT columns                         |      13.371 |         1.913 |
| + own polynomial sincos                       |      12.234 |         1.862 |
| + floor-based phase fold                      |      11.805 |         1.608 |
| + AVX2 evolve                                 |   **8.378** |     **1.426** |

**16.6x overall.** A 512x512 ocean now updates in 1.43 ms, leaving 15.2 ms of a
60 Hz frame for everything else.

### Where the time goes now

At 512 squared serial: **FFT 7.00 ms (84%)**, evolve 0.45 ms (5%), finalise
0.62 ms (7%). The FFT is dominant again.

The remaining FFT opportunity is specific and measured: the row pass still runs
its first three stages scalar, because there the memory stride `s` is 1, 2 and
4, all below the 8-float vector width. That is 3 of 9 stages at N = 512, so
roughly a third of the row-pass butterflies. The column pass does not have this
problem, because batching eight columns multiplies the stride by eight.

Two ways to fix it, neither yet attempted:

1. A **radix-8 first stage**, folding s = 1, 2 and 4 into one pass that is
   vectorisable by construction.
2. A **blocked transpose** before the row pass, so rows can be batched the same
   way columns already are. Costs two extra passes over memory, which may eat
   the gain - it would have to be measured.

Finalisation at 7% is memory-bound (it writes 8 MB per frame at 512 squared)
and is not worth vectorising.

---

## Simulated low-end hardware

Measured 2026-09-18, same machine (i7-14700HX), using `--simd` and `--threads`
to force a weaker configuration rather than needing weaker hardware. Same
settings as above (patch 200 m, wind 12 m/s, choppiness 1.0).

| configuration                          | 256x256 update | vs. native | 512x512 update | vs. native |
|-----------------------------------------|----------------:|-----------:|-----------------:|-----------:|
| native (AVX2, 28 threads)               |         0.492 ms |       1.0x |          1.692 ms |       1.0x |
| AVX2, 4 threads (old quad-core)         |         0.557 ms |       1.1x |          3.818 ms |       2.3x |
| SSE2, 2 threads (budget dual-core)      |         1.957 ms |       4.0x |          8.980 ms |       5.3x |
| **scalar, 1 thread (circa-2008 laptop)**|     **5.290 ms** |  **10.8x** |     **26.396 ms** |  **15.6x** |

Run with `ocean_bench.exe --simd <scalar\|sse2\|avx2\|neon\|max> --threads N`.

### Reading this honestly

This is a **lower bound** on the real-world slowdown, not a prediction of
actual performance on old hardware. It measures the algorithmic cost of
dropping SIMD width and core count on *today's* clock speed, cache sizes and
memory bandwidth. A real 2008 CPU is also clocked lower and has a much smaller
cache, both of which cost more on top of what is shown here.

What it does tell us reliably: at 256x256, even the worst case fits inside a
60 Hz budget (5.29 ms of 16.7 ms). At 512x512 it does not (26.4 ms) - a
low-end target should either stay at 256x256 or accept a lower update rate for
the ocean specifically (it does not need to update every rendered frame; see
the V2 plan).

### The determinism claim, demonstrated end to end

The unit tests already prove the *output buffers* are bit-identical across
SIMD levels and thread counts (ADR-012, ADR-013, ADR-016). To check this holds
all the way through the renderer too - not just the library's own buffers -
the viewer's `--screenshot` mode was run twice with identical scene settings:
once at native speed (AVX2, 28 threads) and once forced to the worst case
(scalar, 1 thread), and the two captured frames were byte-compared.

**Result: 0 differing bytes across a 4 320 054-byte capture.** The rendered
image is pixel-for-pixel identical despite an 11-16x difference in how long it
took to compute. That is the promise made concrete: a game running on a weak
CPU shows the exact same ocean as one on a fast CPU, just less often.

This required a real fix first: `--screenshot` was advancing simulation time
by measured wall-clock `dt`, so the same `--frames N` reached a different
simulated time depending on how fast the machine rendered the warmup frames -
making two screenshots from different configurations incomparable by
construction, independent of anything about the library itself. Screenshot
mode now advances by a fixed 1/60 s virtual timestep instead, which also makes
it usable for visual regression testing across machines generally.

---

## NEON verification (QEMU aarch64 emulation)

Measured 2026-09-18. No physical ARM hardware was used - see ARCHITECTURE.md
ADR-018 for the full methodology and its limits. Cross-compiled with
`aarch64-linux-gnu-g++` 13.3.0, run under `qemu-aarch64-static` 8.2.2 inside
WSL2 Ubuntu 24.04, statically linked, `-O3`.

```
detected SIMD level: NEON
test cases:     91 |     91 passed | 0 failed | 0 skipped
assertions: 633425 | 633425 passed | 0 failed |
```

The FFT NEON stage kernel matched the scalar reference bit for bit across
every stage, stride and size the test suite checks - the same standard held
for the AVX2 and SSE2 kernels on x86. Reproducible via
`scripts/verify-neon-qemu.sh`.

---

## Shallow-water dispersion: no per-frame cost

The finite-depth dispersion relation (ADR-019) is evaluated only when
building the spectrum tables at construction time - the same point the
existing deep-water sqrt(g*k) was always computed. It adds one std::tanh
call per grid cell, once, not per frame; evolve_rows and finalize_rows,
the two functions actually on the per-frame path, are untouched.

Re-measured 2026-09-18 after the change, deep water (default), same
machine and settings: 512x512 threaded 1.232 ms - consistent with (in
fact slightly better than, within normal laptop thermal variance) the
1.426 ms recorded after the AVX2 evolve kernel above. No regression.

---

## Cascades (ADR-020)

Measured 2026-09-18, i7-14700HX, 256x256 per level, patches 800/150/25 m
(far/mid/near), wind 12 m/s.

| configuration                       | update time | notes |
|--------------------------------------|------------:|-------|
| single Ocean::update() (256^2)      |    0.512 ms | baseline |
| CascadeStack, 3 levels (256^2 each) |    1.549 ms | 3.02x - linear, as expected |
| CascadeStack::height_at()           |   0.400 us  | 3 independent fixed-point solves |

Scaling is linear because update() runs levels sequentially; each level
already parallelises internally (own thread pool or shared host scheduler).
1.55 ms total leaves comfortable room in a 60 Hz (16.7 ms) budget even before
the renderer's own cost.

Verification highlight: for three statistically independent, spectrally
non-overlapping cascades, the realised composite variance matched the sum of
each level's own variance to within 8% (Monte Carlo sampling noise plus
expected small overlap at each cutoff's soft knee) - Var(a+b+c) = Var(a) +
Var(b) + Var(c) for independent fields, verified on real generated data
rather than assumed.

---

## Interaction field (ADR-021)

Measured 2026-09-18, i7-14700HX, AVX2, cell size held at 0.25 m so the kernel
and timestep are identical across sizes. One `update()` at 60 fps runs exactly
one substep, so these are also per-frame costs at 60 fps.

Run it with `build/bench/ocean_bench_interaction.exe`.

### Against grid size, P = 6 (169 taps)

| N     | serial (ms) | threaded (ms) | speedup | Mcell/s | dt limit (s) |
|------:|------------:|--------------:|--------:|--------:|-------------:|
| 128²  |       0.111 |         0.116 |   0.96x |   140.8 |       0.1539 |
| 256²  |       0.476 |         0.191 |   2.48x |   342.4 |       0.1539 |
| 512²  |       1.894 |         0.350 |   5.41x |   748.8 |       0.1539 |

Serial scales as 4.0x per 4x the cells, which is what a stencil whose work is
exactly proportional to cell count should do.

### Against kernel radius, 256²

| P  | taps | serial (ms) | threaded (ms) | ms/Mtap | dispersion error |
|---:|-----:|------------:|--------------:|--------:|-----------------:|
|  3 |   49 |       0.178 |         0.133 |  0.0553 |           36.08% |
|  4 |   81 |       0.248 |         0.149 |  0.0467 |           26.53% |
|  6 |  169 |       0.462 |         0.185 |  0.0417 |           13.11% |
|  8 |  289 |       0.768 |         0.201 |  0.0406 |            6.47% |
| 10 |  441 |       1.298 |         0.238 |  0.0449 |            3.03% |

Dispersion error is the peak symbol error over the band the kernel claims; the
wave-**speed** error is half of it.

**The threaded column makes a case for P = 8 that the serial column hides.**
Serially, P = 8 costs 1.66x P = 6 — which is just the tap ratio 289/169 = 1.71,
so the loop is doing exactly what it should. Threaded, it costs only **8.6%
more** (0.201 against 0.185 ms) while halving the dispersion error, because at
256² the dispatch overhead dominates and there is spare parallel capacity to
absorb the extra taps. On this machine P = 8 is close to free; the default
stays at 6 because that is not true of a machine with fewer cores, where the
serial column is the honest guide.

### Next to the FFT ocean

| N     | ocean (ms) | interaction (ms) |
|------:|-----------:|-----------------:|
| 128²  |      0.357 |            0.156 |
| 256²  |      0.539 |            0.200 |
| 512²  |      1.342 |            0.327 |

At the default 256², the interaction field costs about **37% of one
`Ocean::update()`** and roughly 13% of a 3-level cascade frame (1.549 ms,
measured in the cascades section above).

### A 6.8x speedup found by an impossible number

The first run of this benchmark reported 256² as **four times slower per cell
than 512²**. That is not a slow path, it is an impossible one: the work is
exactly proportional to cell count, so per-cell cost cannot fall as the grid
grows.

Chasing it ruled out substep count (1 everywhere, confirmed by instrumenting
the loop) and cache capacity (which would make the larger grid slower, not
faster). What actually differed was **how many substeps each size had run** by
the time it was measured — 400 iterations at 256², 120 at 512², 1200 at 128².

The absorbing layer multiplies its cells by about 0.8 every substep. After a
few hundred steps a ring of cells is sitting in the **denormal** range, where
x86 arithmetic traps to microcode at roughly 100x the cost. 512² had not yet
decayed into that range; 128² had already passed through it to exact zero;
256² was sitting in it for the whole measurement.

Flushing values below 1e-30 to exactly zero at the end of each substep:

| N     | before (ms) | after (ms) | speedup |
|------:|------------:|-----------:|--------:|
| 128²  |       0.147 |      0.124 |   1.20x |
| 256²  |       1.325 |      0.196 | **6.76x** |
| 512²  |       2.098 |      0.340 | **6.17x** |

The alternative fix, setting FTZ/DAZ in MXCSR, was rejected: it is a
process-wide CPU mode that would change results in the host's own code, and it
is an x86 register with no portable equivalent, so the scalar, SSE2, AVX2 and
NEON paths would stop agreeing bit for bit and ADR-013's contract would quietly
break. An explicit compare-and-zero is portable, deterministic and identical on
every path — and it is inside the bit-exactness test like everything else.

This is the second time in this project that a benchmark caught a bug rather
than merely reporting a speed (the first was the signed-zero AVX2 evolve bug in
ADR-016). Both were found the same way: by noticing a number that could not be
true and refusing to move on.

### Threading threshold

ADR-012 measured 8192 cells for the FFT pipeline. This workload is different —
one dispatch per substep rather than four barriers, but far less work to spread
at small sizes — and it has its own crossover: 128² (16384 cells) measured
**slower** threaded, 0.137 ms against 0.112 ms serial. Only powers of two are
legal sizes, so the crossover can only be located to (16384, 65536]; the
threshold is set to 32768 between them, which brings 128² back to parity
(0.96x). The principle is ADR-012's and is the defensible part: threading must
never make things worse.

### Orbital velocity

Measured 2026-09-18, i7-14700HX, AVX2, threaded. `OceanDesc::compute_velocity`
is off by default; an integration that does not ask for it pays nothing.

| N     | ocean (ms) | with velocity (ms) | ratio |
|------:|-----------:|-------------------:|------:|
| 128²  |      0.390 |              0.436 | 1.12x |
| 256²  |      0.554 |              0.664 | 1.20x |
| 512²  |      1.351 |              2.000 | 1.48x |

Three extra real spectra pack into two more complex fields, so the frame runs
**six** inverse transforms instead of four. 6/4 = 1.5x on the FFT stage alone,
which is what 512² converges to as the transforms come to dominate; at smaller
sizes the fixed costs dilute it.

The velocity spectra are computed in a **second pass** over the same rows
rather than as an extension of `evolve_rows`. That costs a second argument
reduction and `sincos` pair per cell, and it means the velocity half is
currently scalar while the base half stays AVX2 - but it leaves the existing
evolve kernel completely untouched, so switching velocity on cannot perturb a
single bit of the surface an existing integration already renders. A test
asserts exactly that with `memcmp` across four times.

Fusing the two passes and writing the AVX2 velocity kernel is the obvious
optimisation. It is deliberately not done yet: the standing practice here is to
measure before optimising, and nothing has yet shown a real scene limited by
this.

### Query cost

Measured 2026-09-18, i7-14700HX, 256 squared ocean plus a 256 squared
interaction field, 20000 queries at non-grid-aligned positions.

| query                                | microseconds |
|--------------------------------------|-------------:|
| `Ocean::sample_at`                   |       0.1107 |
| `InteractionField::sample_at`         |       0.0022 |
| `WaterSurface::sample_at` (combined) |       0.1146 |
| **added cost of the interaction**    |   **0.0039** |

**The combined query costs 3.5% more than the FFT query alone.** The reason is
structural: the interaction field is a pure height field, so it needs one
bilinear tap and nothing else. The FFT query has to invert the choppy
displacement first (ADR-011's fixed-point solve, four iterations of bilinear
fetches), and that inversion is untouched by adding a height field on top of
it - there is no second solve to do.

At 0.115 microseconds a host can afford roughly 8500 combined queries per
millisecond, so a few hundred buoyancy probes per frame is not a cost worth
thinking about.

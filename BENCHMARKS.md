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

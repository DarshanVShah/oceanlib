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

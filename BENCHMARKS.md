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
| 64²   |      0.243 |     0.237 |       0.084 |    0.152 |         0.009 |    16.9 |
| 128²  |      1.171 |     1.136 |       0.344 |    0.770 |         0.037 |    14.0 |
| 256²  |      4.998 |     4.838 |       1.304 |    3.461 |         0.153 |    13.1 |
| 512²  |     23.620 |    22.896 |       5.346 |   17.176 |         0.634 |    11.1 |

Stage columns are measured separately, each with its own warmup, so they do not
sum exactly to the total.

### What this says

**The FFT dominates: ~73% of frame time at 512².** Four inverse 2D transforms,
each N row transforms plus N column transforms. This is where SIMD effort
belongs first.

**Spectrum evolution is ~23%.** That stage is one `sinf`/`cosf` pair per cell —
262 144 of them at 512² — plus the eight spectrum derivations. A vectorised
sine/cosine is the obvious lever, and it would have a second benefit: a
polynomial approximation we write ourselves is bit-identical across platforms,
which would close the last cross-runtime determinism gap noted in ADR-003.

**Finalisation is ~3%** and is not worth optimising. It is memory-bound,
writing 8 MB per frame at 512², and already close to what the store bandwidth
allows.

**Scaling.** Cell throughput falls from 16.9 to 11.1 Mcell/s as N grows, which
is the expected shape: per-cell work grows as `log N` from the extra FFT
stages, and the working set stops fitting in cache (at 512² the four complex
fields alone are 8 MB, past this CPU's 33 MB L3 only in aggregate but well past
L2).

**Where this leaves us.** 256² at 5.0 ms fits a 60 Hz budget single-threaded
with room to spare. 512² at 23.6 ms does not — it does not even fit 30 Hz
comfortably once a renderer needs time too. Threading and SIMD are what close
that gap, and the numbers above are the baseline they will be measured against.

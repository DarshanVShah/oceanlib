# oceanlib — Benchmark results

Only measured numbers go in this file. Nothing here is estimated or projected.

## Test machine

| Field    | Value                                                       |
|----------|-------------------------------------------------------------|
| CPU      | Intel Core i7-14700HX (8 P-cores + 12 E-cores, 28 threads)  |
| ISA      | SSE2 / AVX2 available; **no** AVX-512 (fused off on Raptor Lake) |
| OS       | Windows 11 Home 26200                                       |
| Compiler | MSVC 19.44.35208 (VS 2022 17.14), x64                       |
| Flags    | `/O2 /fp:precise` (Release), no fast-math                   |

## Results

_No benchmarks recorded yet — the FFT and update loop do not exist. This
section is filled in as `bench/` targets land._

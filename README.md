# oceanlib

A small, dependency-free, engine-agnostic C++20 library for real-time ocean
waves: a JONSWAP directional spectrum, an in-tree multithreaded SIMD FFT, and
CPU-side displacement / normal / foam buffers any renderer can upload.

**Status: early development.** See [ARCHITECTURE.md](ARCHITECTURE.md) for the
design and the reasoning behind every decision.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Options: `OCEAN_BUILD_TESTS` (ON), `OCEAN_BUILD_BENCH` (ON),
`OCEAN_BUILD_EXAMPLES` (OFF). With tests off, the build needs no network and
no external dependency of any kind.

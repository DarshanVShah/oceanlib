# oceanlib

A small, dependency-free, engine-agnostic C++20 library for real-time ocean
waves: a JONSWAP directional spectrum, an in-tree multithreaded SIMD FFT, and
CPU-side displacement / normal / foam buffers any renderer can upload.

![Calm sea, 12 m/s wind](docs/ocean-calm.png)

512x512 waves update in **1.43 ms** on an i7-14700HX - 16.6x faster than
the first working version. Above: the included
Vulkan viewer, 256x256 grid tiled 7x7, 6.4M triangles. The sun glitter is not
an effect - it falls out of the wave slopes in the normal buffer.

![Whitecaps at 22 m/s](docs/ocean-foam.png)

At 22 m/s with heavier chop, foam appears along the crests. It is driven by the
Jacobian of the horizontal displacement - the actual measure of the surface
folding onto itself - not by a height threshold.

## Why it exists

Four promises, and every one of them is tested rather than asserted:

1. **Runs on anything.** A multithreaded, SIMD-optimised CPU FFT is the
   baseline, so it works on hardware with no compute shaders. Scalar, SSE2,
   AVX2 and NEON kernels ship in one binary and are selected by CPUID - and
   they are **bit-identical to each other**, so the ocean never depends on
   which CPU drew it.
2. **Physics matches visuals.** `height_at(x, z)` returns exactly what the
   renderer displays, with no GPU readback. Because choppy displacement moves
   vertices sideways, that is a fixed-point inversion rather than a lookup:
   measured mean error 0.01 mm against 21.8 mm for a naive direct lookup.
3. **Deterministic.** Same seed and time give byte-identical buffers: across
   runs, across thread counts, across SIMD levels, and across platforms - the
   library ships its own PRNG, its own normal distribution and its own
   sine/cosine precisely so that nothing depends on a vendor's libm.
4. **Easy to integrate.** The output buffers are laid out as RGBA32F textures,
   so uploading the whole ocean is two `memcpy`s and two image copies.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

Options: `OCEAN_BUILD_TESTS` (ON), `OCEAN_BUILD_BENCH` (ON),
`OCEAN_BUILD_EXAMPLES` (OFF), `OCEAN_BUILD_VIEWER` (OFF).

With tests off, the build needs no network and no external dependency of any
kind. doctest is fetched only for tests; Vulkan and GLFW are needed only for
the viewer.

## Using it

```cpp
#include "ocean/ocean.hpp"

ocean::OceanDesc desc;
desc.size                = 256;
desc.patch_length        = 200.0f;   // metres per tile
desc.spectrum.wind_speed = 12.0f;    // m/s
desc.seed                = 1337;

ocean::Ocean sim{desc};              // everything allocates here, once

// per frame - performs no heap allocation
sim.update(absolute_time_seconds);

const ocean::Buffers b = sim.buffers();
// b.displacement : N*N*4 floats, (dx, dy, dz, foam)
// b.normal       : N*N*4 floats, (nx, ny, nz, jacobian)
// Both are exactly RGBA32F textures. Upload and sample.

// Physics, agreeing exactly with what is drawn:
float water_height = sim.height_at(boat_x, boat_z);
```

A C API is in `include/ocean/ocean.h` for bindings from other languages.

To plug in an engine's scheduler instead of the built-in thread pool, set
`desc.parallel_for` to anything with the shape of `ParallelFor` /
`IJobParallelFor` + `Complete()` / `tbb::parallel_for`.

## The viewer

```sh
cmake -S . -B build -DOCEAN_BUILD_VIEWER=ON
cmake --build build
./build/examples/viewer/ocean_viewer
```

Needs the Vulkan SDK (for headers and `glslc`); GLFW is fetched automatically.

| key | action | key | action |
|-----|--------|-----|--------|
| WASD / QE | move | mouse + RMB | look |
| Shift | move faster | Tab | wireframe |
| Space | pause time | R | reset camera |
| 1 / 2 | choppiness | Esc | quit |

Flags: `--size N --mesh M --tiles T --wind U --chop C --foam F --wireframe
--screenshot out.bmp --frames N`.

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - every design decision and why,
  including the alternatives rejected and the bugs each choice avoids.
- **[BENCHMARKS.md](BENCHMARKS.md)** - measured numbers only, with the CPU,
  compiler and flags they were taken on.

## Status

Version 1 core is complete: spectrum, FFT, threading, SIMD, queries, C API,
and the Vulkan viewer. Not yet built: cascades (multiple overlapping patch
sizes), shallow-water dispersion, GPU compute, and time-looped baking.

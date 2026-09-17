# oceanlib — Architecture & Decision Log

A small, dependency-free, engine-agnostic C++20 library for real-time ocean
waves. This document records every architectural decision and the reasoning
behind it, newest sections appended as we go.

## Core promises

1. **Runs on anything.** Multithreaded, SIMD-optimised CPU FFT is the baseline,
   so the library works on hardware without compute shaders. A GPU path may be
   added later but is never required.
2. **Physics matches visuals.** Height and normal queries at any world position
   return exactly what the renderer displays, with no GPU readback.
3. **Deterministic.** Same seed + same time => identical waves on every run.
4. **Easy to integrate.** Outputs plain CPU buffers (displacement, normals,
   foam) that any renderer or engine can upload.

## Layout

| Path       | Contents                                                   |
|------------|------------------------------------------------------------|
| `include/` | Public API. Anything here is a compatibility commitment.    |
| `src/`     | Implementation + internal headers (`src/core/...`).         |
| `tests/`   | doctest suite; the only place an external dep is allowed.   |
| `bench/`   | Benchmark executables.                                     |
| `examples/`| Small integration samples.                                 |

Internal headers deliberately live under `src/`, not `include/`. If a header
is not in `include/ocean/`, users cannot accidentally depend on it, so we stay
free to change it.

---

## ADR-001 — Build system: CMake, C++20, zero runtime dependencies

**Decision.** CMake >= 3.20, C++20, a single `ocean` static library target with
an `ocean::ocean` alias. doctest is pulled in by `FetchContent` *inside*
`tests/`, so it never enters the library's dependency graph.

**Why.** CMake is what every engine's build already understands, and an alias
target means downstream `find_package` and `add_subdirectory` usage look
identical. Keeping the fetch inside `tests/` is what actually enforces promise
#4 — a user who builds with `-DOCEAN_BUILD_TESTS=OFF` needs no network at all.

**Alternatives considered.** Vendoring doctest in-tree (no network needed, but
a multi-megabyte blob in a library whose selling point is smallness); Catch2
(heavier compile times for no benefit at this size).

---

## ADR-002 — Floating-point policy: strict IEEE, never fast-math

**Decision.** MSVC builds with `/fp:precise`; GCC/Clang with `-fno-fast-math
-ffp-contract=off`. No exceptions, including in the SIMD paths.

**Why.** Fast-math lets the compiler reassociate float arithmetic
(`(a+b)+c -> a+(b+c)`) and fuse multiply-add pairs. Both change results in the
low bits, and both are applied *opportunistically*, so the output starts
depending on compiler version, optimisation level and which instruction set
the vectoriser picked. That directly violates promise #3.

`-ffp-contract=off` matters specifically because FMA contraction is on by
default in GCC/Clang even without fast-math. An FMA keeps more precision than
the separate mul+add that the scalar reference path performs, so the SIMD and
scalar FFTs would disagree by more than rounding — and our "optimised FFT
matches scalar FFT" test would silently be comparing two genuinely different
computations.

**Cost.** We give up some auto-vectorisation of the scalar paths. We accept
that, because the paths that matter get hand-written SIMD anyway.

---

## ADR-003 — Determinism: own PRNG *and* own normal distribution

**Decision.** PCG32 (`src/core/rng.hpp`) plus a hand-written Box-Muller
transform. `std::normal_distribution` is banned.

**Why.** The C++ standard pins the *engines* (`std::mt19937` is bit-exact
everywhere) but leaves *distributions* entirely unspecified. libstdc++, libc++
and MSVC all implement `std::normal_distribution` differently — different
algorithms, different internal caching of the second Box-Muller output — so
the same engine and the same seed give different numbers on each. Promise #3
would fail at the first platform boundary.

**Why PCG32 specifically.** 8 bytes of state (vs 2.5 KB for mt19937, which is
cache-hostile when we are streaming a spectrum), BigCrush-quality output, and
a clean multi-stream design: any odd increment defines a distinct period-2^64
stream. That last property is what will let us give independent parts of the
simulation their own streams and keep bit-identical results regardless of
thread scheduling.

**Why plain Box-Muller over Ziggurat / Marsaglia polar.** Both of the faster
algorithms use *rejection*, so the number of engine draws consumed per output
varies with the values drawn. Box-Muller always consumes exactly two draws and
emits exactly two normals, which makes stream position a pure function of
sample index — we can generate any tile of the spectrum from any thread in any
order and get identical bits. It also happens to fit the physics exactly:
Tessendorf's initial amplitude h0(k) needs precisely two independent Gaussians
(real and imaginary parts) per grid cell, i.e. one Box-Muller pair, so nothing
is ever cached or wasted. The transform costs a `log`, `sqrt` and `sincos`,
but it runs only at init, never per frame.

**Known limit, stated honestly.** This gives bit-identical results run-to-run
on a given build, and identical *algorithms* across platforms. It does not
guarantee bit-identity across different C runtimes, because `std::log`,
`std::cos` and `std::sin` are permitted to differ by an ULP between libm
implementations. Closing that last gap needs our own software transcendentals;
the error is ~1e-16 relative and invisible in the wave field, so we have not
paid that cost. If exact cross-platform reproducibility ever becomes a hard
requirement (e.g. lockstep multiplayer that reads ocean height in gameplay),
this is the one place to revisit.

**Verification.** `tests/test_rng.cpp` holds a known-answer test against the
canonical PCG32 outputs for seed 42 / stream 54, re-derived independently from
the algorithm definition rather than captured from our own output.

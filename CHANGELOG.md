# Changelog

All notable changes to oceanlib are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[semantic versioning](https://semver.org/).

## [1.0.0] - 2026-09-17

First public release. Everything below was already built and tested
pre-1.0; this entry is the complete surface being committed to.

### Added

- **Spectrum.** JONSWAP directional wave spectrum with Hasselmann
  directional spreading, wind sea through narrow swell (`SpectrumDesc::swell`),
  and an optional finite-depth dispersion relation for shallow/coastal
  water (`SpectrumDesc::depth`).
- **Simulation core.** In-tree complex FFT (own implementation, validated
  against a naive O(N^4) DFT), a three-stage per-frame pipeline
  (evolve / transform / finalise), and Tessendorf choppy displacement.
- **SIMD.** Scalar, SSE2, AVX2 and NEON kernels selected at runtime by
  CPUID, shipped in one binary, and **bit-identical to each other** -
  enforced by a test that `memcmp`s every kernel against the scalar
  reference at every stage and size. NEON verified under aarch64 QEMU
  emulation (no physical ARM hardware used; see ARCHITECTURE.md ADR-018).
- **Threading.** A pluggable blocking parallel-for hook
  (`OceanDesc::parallel_for`) that maps onto Unreal's `ParallelFor`,
  Unity's `IJobParallelFor`, `tbb::parallel_for`, or a plain pool; falls
  back to a built-in thread pool when unset.
- **Queries.** `height_at()` / `sample_at()` invert the choppy
  displacement via fixed-point iteration to answer "what does the
  renderer actually draw here", rather than a lookup that ignores chop -
  measured mean error 0.01 mm against 21.8 mm for a naive lookup at
  default choppiness.
- **Cascades.** `ocean::CascadeStack` sums multiple independently-scaled
  Ocean patches into one composite sea state (e.g. far swell + mid chop +
  near ripple), with height and horizontal offset as exact linear
  superposition. C++ only for this release; a C API is planned for v1.1.
- **C API.** `include/ocean/ocean.h` - an opaque-handle wrapper with no
  exceptions crossing the C boundary, and `struct_size`-based descriptor
  versioning so a shared library can gain fields without breaking older
  callers.
- **Determinism.** Own PRNG (PCG32), own Box-Muller normal distribution,
  and own polynomial sine/cosine, specifically so that the same seed and
  time give byte-identical output across runs, thread counts, SIMD
  levels, and platforms - never dependent on a vendor's libm.
- **Vulkan reference viewer** (`examples/viewer`), demonstrating buffer
  upload, cascades, and a `--screenshot` mode used for cross-configuration
  visual regression checks.
- **Documentation.** ARCHITECTURE.md (every design decision, the
  alternatives rejected, and the bugs each choice avoids) and
  BENCHMARKS.md (measured numbers only, with the machine and flags they
  were taken on).
- MIT license.

### Performance

512x512 `Ocean::update()`: 23.62 ms (scalar, single-threaded) to 1.43 ms
(AVX2, threaded) on an i7-14700HX - 16.6x. Full breakdown and methodology
in BENCHMARKS.md.

[1.0.0]: https://github.com/DarshanVShah/oceanlib/releases/tag/v1.0.0

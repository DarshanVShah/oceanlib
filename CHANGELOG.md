# Changelog

All notable changes to oceanlib are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[semantic versioning](https://semver.org/).

## [Unreleased] - v3 work in progress

Local interaction: objects disturb the water, the disturbance propagates and
reflects, and gameplay queries see it. See ARCHITECTURE.md ADR-021.

### Added

- **Interaction field** (`ocean/interaction.hpp`). A local height field evolved
  by the iWave convolution (Tessendorf, *Game Programming Gems 4*, 2004) and
  added on top of the FFT surface. Configurable resolution and extent,
  recentring, damping, an absorbing boundary layer and an obstruction mask.
  Kept as a SEPARATE object from `Ocean` on purpose: `Ocean::update()` takes
  absolute time and is a pure function of `(seed, desc, time)`, and an
  interaction field is a time-stepped ODE that cannot be.
- **Disturbance API.** Immediate-mode queue of impulses and continuous sources
  in world space, with position, radius, strength and velocity. No handles, so
  nothing to own across the C boundary. Submitting never allocates; overflow is
  counted, not silently dropped.
- **`WaterSurface`.** Non-owning combiner over an `Ocean` or `CascadeStack`
  plus an `InteractionField`, so one query returns the surface including the
  wake. Normals compose through slopes, which is exact.
- **3-D orbital velocity** (`OceanDesc::compute_velocity`, off by default).
  Three new spectra derived by solving the dynamic boundary condition per
  branch; six inverse transforms instead of four when enabled.
- **C API** for both: `ocean_interaction_*`, `ocean_surface_ex`,
  `ocean_sample_ex`, `ocean_get_velocity`, `ocean_sample_combined`.
- **Viewer.** Left click ray-marches against the displaced surface and drops a
  rock there. Debug controls for impulse strength and radius, and a toggle that
  isolates the interaction field from the swell. New flags for reproducible
  captures: `--splash`, `--isolate`, `--impulse`, `--impulse-radius`,
  `--interaction`, `--cam-x/y/z`, `--cam-yaw`, `--cam-pitch`.
- **`bench/ocean_bench_interaction`** and a new BENCHMARKS.md section.

### Changed

- `ocean::Surface` gains `velocity_x/y/z`. Source-compatible; the C
  `ocean_surface` is deliberately frozen because it is returned by value, and
  velocity is exposed through `ocean_surface_ex` instead.
- `ocean_desc` gains `compute_velocity`, appended at the true end so
  `struct_size` stays meaningful.
- Interaction threading threshold is 32768 cells, not ADR-012's 8192; that
  figure was measured for the FFT pipeline and this workload has its own
  crossover.

### Fixed

- **A 6.8x denormal stall.** The absorbing layer decays cells geometrically, so
  after a few hundred substeps a ring of them sits in the denormal range where
  x86 arithmetic traps to microcode. Found because 256 squared measured four
  times slower *per cell* than 512 squared, which is impossible. Values below
  1e-30 are now flushed to zero, in all four kernels, bit-identically.
- **A clipped Ricker impulse did not conserve volume.** The continuous profile
  integrates to exactly zero but a cell-sampled copy clipped at 3 sigma does
  not, and net volume is the one thing this operator can never propagate away.
- **`recenter()` left the public buffer stale**, so a query between a recentre
  and the next update read the old field through the new origin.
- **The viewer sampled the interaction field per vertex**, on a mesh far too
  coarse to carry the ripples, so they were aliased away entirely.
- **The viewer recentred the field on the camera** rather than on what the
  camera was looking at, so impulses aimed at the visible water fell outside
  the grid.
- **`compute_velocity` never reached `translate()`** in the C API, so a C
  caller could set it and silently get no velocity.

### Known limitations

- The interaction field contributes VERTICAL velocity only. Horizontal orbital
  velocity would need the Riesz transform of `dEta/dt` - two more convolutions
  per substep.
- The velocity spectra are computed in a scalar second pass; the AVX2 velocity
  kernel is not written. Measured cost is 1.12x to 1.48x.
- Benchmarks are from one machine (i7-14700HX).
- The kernel is accurate over roughly 2 to 16 cells of wavelength. Long waves
  are the FFT ocean's job; this is a property of any finite even stencil, not
  of this implementation.

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

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

---

## ADR-004 — Output layout: two GPU-texture-shaped RGBA float4 buffers

**Decision.**

```
displacement[4*i + 0..2] = dx, dy, dz      normal[4*i + 0..2] = nx, ny, nz
displacement[4*i + 3]    = foam            normal[4*i + 3]    = jacobian
```

Row-major, `i = z * N + x`, both 64-byte aligned, `N*N*4` floats each.

**Why.** This is exactly an `RGBA32F` texture, so uploading the entire ocean is
two `glTexSubImage2D` / `UpdateSubresource` calls with no repacking pass on the
user's side. That is promise #4 in concrete form.

The obvious objection is that planar SoA is friendlier to SIMD, and internally
that is true — so internally we keep planar scratch, and the FFTs write planar.
The interleave is *free* because a finalisation pass has to happen anyway:
normals must be normalised, and foam must be derived from the Jacobian
determinant of the displacement gradients. That pass reads planar scratch and
writes the interleaved form in the same sweep, so interleaving costs no extra
traversal of memory.

We also get the 4th lane for free. A 3-float texel is not a thing GPUs like;
padding to 4 would waste the slot, so we spend it on foam and the raw Jacobian
— data a renderer genuinely wants and would otherwise need a third buffer for.

**Alternatives considered.** Seven planar arrays (28 B/texel, no padding waste,
best SIMD — but seven uploads, or the user writes our interleave pass for us).
A single 28-byte interleaved vertex struct (drops into a vertex buffer, but a
28-byte stride is never 16-byte aligned and matches no GPU texture format).

**Why 64-byte alignment.** Not for the loads — AVX2 handles unaligned fine —
but for false sharing. Threads write neighbouring tiles of the same buffer; if
two tiles shared a cache line, that line would ping-pong between cores. Aligning
the base and sizing tiles in whole lines removes that failure mode.

---

## ADR-005 — Public API: RAII class with pimpl

**Decision.** `class Ocean` holding `std::unique_ptr<Impl>`. Move-only. The
constructor validates and allocates; `update()` never allocates.

**Why.** C++ users get destructors and exception safety, and the public header
pulls in only `<cstddef> <cstdint> <memory>` — no FFT plan, no thread pool, no
spectrum tables. That keeps ABI stable across internal changes and keeps
compile times down for every translation unit that includes us. The C wrapper
will `reinterpret_cast` an opaque `ocean_sim*` straight to `Ocean`, so pimpl
costs the C path nothing.

The usual pimpl objection is the extra indirection. It is one pointer hop per
*API call*, not per sample; an `update()` at 256² touches 65 536 cells, so the
hop is unmeasurable. Exposing the members to avoid it would trade a real
encapsulation benefit for an immeasurable one.

**Error handling.** The constructor throws `std::invalid_argument` on a
malformed descriptor. The C API will catch and return null, so hosts that build
with exceptions disabled use that path. If an exception-free C++ path is ever
needed, the right shape is a static `try_create` returning `std::optional`;
noted, not built.

**`update(time)` takes absolute time, not a delta.** The surface is a pure
function of `(seed, desc, time)` and never of previous calls. Seeking, pausing,
rewinding and replaying all reproduce bit-identically, and a networked client
that joins late gets the same ocean as everyone else. An incremental
`advance(dt)` would accumulate float error in the time variable and make the
state path-dependent — quietly breaking promise #3.

---

## ADR-006 — Threading: a blocking parallel-for callback

**Decision.**

```cpp
using TaskFn        = void (*)(void* ctx, std::uint32_t index);
using ParallelForFn = void (*)(void* user, TaskFn, void* ctx, std::uint32_t count);
```

The host's implementation must run `task(ctx, i)` for all `i` in `[0, count)`
and must not return until every one has finished. A null hook selects the
built-in thread pool.

**Why blocking.** It means we never hold a job handle, so there is no handle
lifetime or ownership question to answer across the C boundary, and no
allocation to track. More importantly the library keeps control of its own
frame ordering: stage boundaries are real barriers, so the result cannot depend
on how the host scheduler interleaved things. Determinism survives contact with
someone else's job system.

It also maps onto every real scheduler unchanged: `ParallelFor` (Unreal),
`IJobParallelFor` + `Complete()` (Unity), `tbb::parallel_for`,
`#pragma omp parallel for`, or a plain pool.

**Alternatives considered.** Submit/wait handles would let us overlap
independent FFT stages, but they force us to define handle lifetime across C,
need an adapter for most engine schedulers, and make deterministic ordering the
caller's problem. Owning our own threads unconditionally is simplest but
oversubscribes cores against the host's job system — precisely the integration
friction we promised to avoid.

**Function pointers, not `std::function`.** A `std::function` may allocate on
copy, and it cannot cross the C ABI. The `void* user` / `void* ctx` pair carries
the state instead, which is the same trick the C API needs anyway.

---

## ADR-007 — "No allocation per frame" is tested, not asserted

`tests/alloc_probe.cpp` replaces the global `operator new` family (plain,
array, nothrow and over-aligned forms) with counting wrappers. The test suite
checks that `update()` moves the counter by exactly zero — and a second test
checks that *constructing* an `Ocean` does move it, so the first test cannot
pass vacuously because the replacement silently failed to link.

This matters because a per-frame allocation is not a small inefficiency: it is
a frame-time spike under lock contention, and on a console with a fixed heap it
is a fragmentation bug that shows up hours in. Code review does not reliably
catch an accidental `std::vector` temporary; a counter does.

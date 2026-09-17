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

---

## ADR-008 — FFT: split storage, Stockham auto-sort, two real fields per transform

### Storage: split `re[]` / `im[]`, not interleaved `Complex{re,im}`

The radix-2 butterfly is

```
t  = w * b                 tr = wr*br - wi*bi
b' = a - t                 ti = wr*bi + wi*br
a' = a + t
```

With split arrays, a SIMD register holds eight *reals* or eight *imaginaries*,
so every lane performs the same operation on the same component. The whole
butterfly is vertical multiplies, adds and subtracts — no shuffles, no
`addsub`, no horizontal operations — and porting between SSE2, AVX2 and NEON
changes only the vector width and the intrinsic spelling.

With interleaved storage a register holds `(r0,i0,r1,i1,…)`, so each complex
multiply needs `moveldup`/`movehdup`-style duplication plus a shuffle and an
`addsub`: roughly three extra operations per butterfly, and a formulation that
does not carry over to NEON at all.

The usual argument for interleaved storage is interoperability — real code
often hands you `std::complex` arrays. That does not apply here: we generate
the spectrum ourselves and we write the interleaved output ourselves, so we
control both ends of the pipeline.

The cost of split storage is two open memory streams instead of one, which
costs a little prefetcher and TLB pressure. At our sizes (N ≤ 1024, so a row is
at most 4 KB per component) both streams sit in L1, so this is not a real cost.

### Algorithm: Stockham auto-sort, not in-place Cooley–Tukey

Textbook Cooley–Tukey permutes the input into bit-reversed order and then runs
`log2(N)` in-place butterfly stages. Two things make that a poor fit here:

1. The bit-reversal pass is a pure scatter — every element moves to an
   unrelated address, with no locality to exploit and nothing to vectorise.
2. The butterfly stride changes every stage. Early stages pair elements that
   are far apart, so loads are gather-shaped rather than contiguous.

Stockham auto-sort removes both. It ping-pongs between two buffers and folds
the permutation into the *write* indices (`2p` and `2p+1`), so there is no
separate reordering pass at all, and within a stage both the read and the write
walk memory sequentially in `q`. That sequential access is precisely what lets
a SIMD version issue full-width loads and stores.

The price is a second buffer — which we were allocating as scratch anyway — and
the fact that the result ends in a different physical buffer depending on the
parity of `log2(N)`. We absorb the parity for free: the final scatter back to
the caller's array reads from whichever buffer the swaps left the data in, so
no copy-back pass is ever needed.

**One twiddle table for all stages.** At the stage with sub-transform length
`len` and stride `s`, we have `len * s == N` at every stage, and the twiddle
for butterfly `p` is `exp(-2πi·p/len)`, which is exactly entry `p*s` of a
single table of `exp(-2πi·j/N)` for `j ∈ [0, N/2)`. So the whole plan needs
`N/2` entries, not one table per stage. The inverse table is the conjugate,
stored separately so the inner loop never spends an instruction negating.

### Strided transforms gather first

A 2D transform is separable, so it is N row transforms followed by N column
transforms. The column pass walks memory with stride N. Rather than thread that
stride through every butterfly, `transform()` gathers the strided input into
contiguous scratch, runs entirely contiguously, and scatters back at the end.
The strided walk therefore happens twice per 1D transform instead of once per
stage — `2` sweeps instead of `log2(N)`.

A bonus verified by test: because the strided and contiguous cases run the
*same* butterfly code on the same contiguous scratch, their results are
bit-identical, not merely close.

### Two real fields per complex transform

Per frame we need eight real fields: height, the two horizontal displacements
`Dx`/`Dz`, two slopes for the normals, and three displacement derivatives for
the Jacobian. Every one of their spectra is Hermitian-symmetric
(`S(-k) = conj(S(k))`), which is exactly the condition for the inverse
transform to be purely real. Running eight complex transforms would mean half
the arithmetic computes imaginary parts that are zero by construction.

By linearity, if `S_a` and `S_b` are both Hermitian then

```
IFFT(S_a + i·S_b) = f_a + i·f_b
```

with `f_a` and `f_b` both real. So one complex transform yields two real
fields: read `f_a` off the real part and `f_b` off the imaginary part. Eight
fields become four transforms — an exact 2× saving, not an approximation, for
about ten lines of packing code.

The alternative that saves the same 2× is a true real-to-complex transform
exploiting Hermitian symmetry to store only `N/2+1` columns. It would also
halve spectrum memory, but it needs its own pre/post twiddle pass, its own
index bookkeeping and its own SIMD path, and symmetry bugs there are subtle.
Packing gets the same speedup with none of that risk. If memory ever becomes
the binding constraint, R2C is the upgrade path.

### Validation strategy

The reference is a **true O(N⁴) quadruple sum**, not a row-column DFT. That
matters: `transform_2d` *assumes* the kernel is separable. Validating it
against a reference that made the same assumption would let a bug in that
assumption cancel on both sides and pass. Summing over all four indices
independently shares no structure with the code under test.

The reference runs in `double` while the FFT runs in `float`, so a disagreement
is unambiguously the FFT's error rather than a contest between two equally
shaky float computations.

Separately, one test pins the **sign convention against theory**, not against
our own reference: the forward transform of `exp(+2πi·k₀j/N)` must be exactly
`N` in bin `k₀` and zero elsewhere. Without it, a sign error in both the FFT
and the reference DFT would cancel and every comparison test would still pass —
while the ocean ran backwards in time.

**Measured accuracy** (MSVC 19.44, `/O2 /fp:precise`, relative to peak
magnitude): 1D rises from 3.6e-8 at N=2 to 1.4e-7 at N=256; 2D from 4.9e-8 at
N=2 to 1.4e-7 at N=64. That is float epsilon (1.19e-7) growing roughly as
√(log₂N), which is the textbook error bound for a radix-2 FFT. Test tolerances
are set at 1e-6 — about 7× the measured error, tight enough that a wrong
twiddle index or flipped sign (both O(1) errors) cannot slip through.

---

## ADR-009 — Spectrum: JONSWAP + Hasselmann spreading, mapped into k-space

### Why JONSWAP rather than Tessendorf's Phillips

Phillips (what Tessendorf 2001 uses) describes a *fully developed* sea: wind has
blown long enough, over enough open water, that the waves have stopped growing.
Real water is usually fetch-limited, and a fetch-limited sea has a sharper,
taller spectral peak than Phillips predicts. JONSWAP adds the peak-enhancement
factor `gamma^r` to model exactly that, and takes **fetch** as a parameter —
which is the knob Phillips simply does not have. It is what separates a young,
steep, short-crested wind sea from mature swell.

Constants used (Hasselmann et al. 1973), with the dimensionless fetch
`x~ = gF/U^2`:

```
alpha   = 0.076 * x~^(-0.22)          omega_p = 22 * (g/U) * x~^(-1/3)
S(w)    = (alpha g^2 / w^5) * exp(-5/4 (w_p/w)^4) * gamma^r
r       = exp(-(w - w_p)^2 / (2 sigma^2 w_p^2))
sigma   = 0.07 below the peak, 0.09 above
```

The asymmetric sigma is not a typo: the measured JONSWAP peak is skewed, rising
more steeply than it falls. Note also that although the `w^-5` equilibrium tail
looks singular as `w -> 0`, the exponential kills it far faster, so the density
goes to zero there. The only guard needed is `w == 0` exactly.

### Directional spreading: the half-angle cosine form

`D(w, theta) = Q(s) * cos^(2s)((theta - theta_w)/2)` with Hasselmann's
frequency-dependent exponent `s(w)`.

Two details that matter:

**The half angle.** `cos^(2s)(theta - theta_w)` would be symmetric front-to-back
and would generate waves travelling *into* the wind carrying the same energy as
waves travelling with it. The half-angle form vanishes at
`theta = theta_w +/- pi`, so upwind waves get zero energy for free — no extra
suppression term is needed.

**The normalisation `Q(s)` cannot be skipped.** It looks like a pure scale
factor, but `s` varies with frequency, so without it the *relative* energy
between frequencies is wrong, not merely the overall level. From
`integral of cos^(2s)(theta/2) dtheta = 2 sqrt(pi) Gamma(s+1/2)/Gamma(s+1)`
over `[-pi, pi]`, we get `Q(s) = Gamma(s+1) / (2 sqrt(pi) Gamma(s+1/2))`,
evaluated through `lgamma` because `Gamma(s+1)` itself overflows a double at
modest `s`.

**Unverified: the swell term.** `s += 16 * tanh(w_p/w) * swell^2` is a
*plausible reconstruction* of Horvath 2015's swell parameter, not a verified
transcription. The character is right (a tanh-weighted boost to the spreading
exponent, strongest below the peak, scaling with the square of the control) but
the constant 16 should be checked against the paper before this is described as
"the Horvath model". It behaves correctly as an artist control regardless.

### The change of variables, which is where the bugs live

Going from `S(omega)` to a 2D wavenumber density needs two changes of variable,
and both are easy to get silently wrong:

1. **Frequency to wavenumber.** `S(k) dk = S(omega) domega`, so multiply by the
   Jacobian `domega/dk = g/(2 omega)` for deep water.
2. **Polar to Cartesian.** `dkx dkz = k dk dtheta`, so
   `Psi(kx,kz) = S(k) D(theta) / k`.

Forgetting the `1/k` is the classic failure. The surface still looks like an
ocean — it just carries the wrong energy balance across scales, so it never
quite matches a real sea state at any wind speed, and no amount of parameter
tuning fixes it.

This is validated directly: a polar quadrature of `Psi` over the whole plane
must reproduce `integral of S(omega) domega`, with both sides truncated at the
same wavenumber. That single test covers the `domega/dk` Jacobian, the `1/k`
factor and the `D` normalisation at once.

### Amplitude normalisation and the factor of 1/2

`h0(k) = (1/sqrt 2)(xi_r + i xi_i) sqrt(P(k))` with `P(k) = Psi(k) dkx dkz / 2`,
which collects to `amp = 0.5 * sqrt(Psi) * dk`.

The `/2` is the part people drop. The time-dependent amplitude is
`h(k,t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt}`, and since `h0(k)` and `h0(-k)`
are independent zero-mean draws, the cross terms vanish in expectation and
`E|h(k,t)|^2 = P(k) + P(-k) = 2 P(k)`. Summing that over the grid must equal the
surface variance, so each bin gets half the density it would otherwise carry.
**Drop the `/2` and the entire ocean comes out sqrt(2) times too tall** — a bug
that looks completely plausible on screen and survives visual inspection
indefinitely. A test compares the realised grid energy against the continuous
prediction to catch exactly this.

### Bin ordering: standard DFT, not centred

Grid index `i` maps to wavenumber index `i` for `i < N/2` and `i - N` above —
standard DFT bin ordering, so the FFT output needs no fftshift. The "centred"
convention `(i - N/2)` would offset the result by half a period, which appears
as a checkerboard sign flip across the grid.

### Per-row RNG streams, via a mixer not a PCG stream

Each row draws from `seed_pcg32(mix64(seed ^ golden*(row+1)), 0)`. Because the
generator is a pure function of `(seed, row)`, rows can be built in any order on
any thread with bit-identical output — the property that makes the build
parallelisable later without touching determinism.

We deliberately do **not** use PCG's `stream` parameter for this. Distinct
streams are guaranteed to be *different* sequences but not statistically
independent; nearby increments are known to produce correlated output, and
correlation between adjacent grid rows would show up as visible banding in the
wave field. A SplitMix64 finaliser scatters adjacent row indices to unrelated
points in state space instead.

### Verification highlights

- Directional spreading integrates to 1 at every frequency tested.
- Polar integral of `Psi` matches `integral of S domega` to 1%.
- **Significant wave height follows the analytically derived fetch-limited
  scaling law.** At fixed fetch, `H_s ~ U^0.886667` (from `alpha ~ U^0.44` and
  `omega_p^-4 ~ U^(4/3)`). Measured across U = 10 -> 20 m/s: **0.8867**. That
  one number validates the exponents inside both `alpha` and `omega_p` plus the
  `omega^-5` tail simultaneously; a single-magnitude check would pass with any
  of them subtly wrong.
- Independent reality check: `H_s = 2.14 m` at U = 10 m/s (about 20 kt,
  Beaufort 5), which is what is actually measured at sea for that wind. This
  catches a units error that a pure scaling test would sail past.
- Note this is far below the `U^2` of a fully developed sea — with fetch pinned
  at 100 km a 20 m/s wind is fetch-limited. Expecting `U^2` is the intuitive
  mistake, and it was the one wrong assertion in the first draft of these tests.

---

## ADR-010 — The per-frame pipeline

Three stages, each splittable by row range so a scheduler can drive them:

1. **`evolve_rows`** — advance every wavenumber to absolute time `t` and pack
   the eight field spectra into four complex fields.
2. **Four inverse 2D FFTs.**
3. **`finalize_rows`** — unpack, build normals, Jacobian and foam, write the
   interleaved output.

### No normalisation on the inverse transform

Tessendorf's surface is literally `h(x) = sum_k h~(k) e^{i k.x}` — the
*unnormalised* inverse sum — and the spectrum amplitudes derived in ADR-009
already carry the correct physical scale. Adding a `1/N^2` here and multiplying
it back out of the spectrum would be pure ceremony. The Parseval test below is
what keeps this honest: a stray `1/N` or `1/N^2` would show up as an `N^2` or
`N^4` discrepancy.

### Phase is reduced in double before narrowing to float

`omega * t` reaches thousands of radians after an hour of simulated time, where
float has only about 1e-4 rad of resolution left. We compute the product in
double, reduce it modulo 2*pi there, and only then narrow for `sinf`/`cosf`.
This is the other half of why `update()` takes absolute time: there is no
accumulator to drift, and seeking to t = 10 hours is exactly as accurate as
seeking to t = 0.1 s.

### Only three displacement gradients, not four

`dDz/dx` equals `dDx/dz`, because the horizontal displacement field is the
gradient of a potential and so its Jacobian matrix is symmetric. Both come out
as `(kx kz/|k|) h~`. That symmetry saves a whole field for free — the reason
eight fields cover everything rather than nine.

### The normal is the exact normal of the *displaced* surface

Parameterising the surface by `(u,v)` as `(u + L*Dx, h, v + L*Dz)` gives
tangents `T_u = (axx, hx, axz)` and `T_v = (axz, hz, azz)` where
`axx = 1 + L*dDx/dx`, `azz = 1 + L*dDz/dz`, `axz = L*dDx/dz`. The normal is
`-(T_u x T_v)`, and its Y component falls out as exactly the Jacobian
determinant, so the normal and the foam term share their work.

The common shortcut is the plain heightfield normal `(-hx, 1, -hz)`. That is
wrong wherever choppiness is doing anything: the vertex has been dragged
sideways, so the real surface there is steeper than the height derivative alone
reports. Shading would disagree with physics, breaking promise #2. Here the
exact form costs nothing extra, because the foam term already needs those same
displacement gradients. With `choppiness == 0` it reduces to `(-hx, 1, -hz)`
exactly, and a test checks that.

### Verification: Parseval is the load-bearing test

For the unnormalised inverse DFT, `mean_x |h|^2 = sum_k |h~(k)|^2`. The DC bin
is exactly zero, so the left side is the variance of the rendered heightfield
and the right side comes straight from the spectrum tables. This is an **exact
identity, not a statistical one**, so it holds to float precision on a single
realisation — measured agreement is within 1e-4 relative.

One test therefore pins down, simultaneously:

- that no stray normalisation crept into the transform,
- that the two-fields-per-transform packing and unpacking are correct (a swap
  would deposit `Dx`'s energy into the height channel),
- that the spectrum really is Hermitian — if it were not, the inverse transform
  would be complex and the real part alone would carry less than the full
  energy.

The reference side is deliberately written out in the test rather than calling
`evolve_rows`, so the thing under test and the thing it is checked against do
not share code.

---

## ADR-011 — World-space queries: inverting the choppy displacement

### The problem

The output buffers are indexed by the **undisplaced parameter position**
`(u,v)`, not by world position. What the renderer actually draws for the cell at
`(u,v)` is the world point

```
X = u + lambda*Dx(u,v),   Y = h(u,v),   Z = v + lambda*Dz(u,v)
```

So the vertex that *lands* at world `(X,Z)` did not start there. Answering
"what is the height at `(X,Z)`?" means inverting that map: solving

```
u + lambda*Dx(u,v) = X
v + lambda*Dz(u,v) = Z
```

for `(u,v)`. There is no closed form — `Dx` and `Dz` are FFT outputs.

This is the whole reason promise #2 needs work at all. With `choppiness == 0`
the map is the identity and a lookup is correct; the moment chop is switched
on, a direct lookup reads the wrong cell.

### The solution: fixed-point iteration

Starting from the undisplaced guess `u = X, v = Z`:

```
u <- X - lambda*Dx(u,v)
v <- Z - lambda*Dz(u,v)
```

**Why it converges.** The iteration map is a contraction exactly while
`|lambda * grad D| < 1` — which is precisely the condition that the Jacobian
determinant stays positive, i.e. that the surface has not folded over itself.
So the solve converges wherever the surface is single-valued, and degrades
exactly where it genuinely is not: inside a breaking wave there really *are*
several surface points above one `(x,z)`, and no solver can choose among them
for us. The failure mode of the algorithm coincides with the failure mode of
the question. That is a good place to be.

Convergence is linear at rate `|lambda * grad D|`, so error falls like
`rate^iterations`. Four steps is the default.

### Measured accuracy

Every vertex of a 128² grid, checked against the height the renderer draws at
that vertex's actual world position. RMS wave height 0.535 m, patch 200 m,
wind 12 m/s:

| choppiness | iterative mean | iterative worst | naive mean | naive worst |
|-----------:|---------------:|----------------:|-----------:|------------:|
| 0.0        | 0.00 mm        | 0.00 mm         | 0.00 mm    | 0.00 mm     |
| 0.5        | 0.00 mm        | 0.03 mm         | 0.00 mm    | 0.00 mm     |
| 1.0        | 0.01 mm        | 1.0 mm          | 21.8 mm    | **725 mm**  |
| 1.5        | 0.06 mm        | 7.7 mm          | 58.7 mm    | 725 mm      |
| 2.0        | 0.25 mm        | 32.6 mm         | 85.6 mm    | 949 mm      |

("naive" = nearest-cell lookup treating the world position as if it were the
parameter position. It reads zero below chop 1.0 only because the displacement
is still under half a cell there and rounds back to the same texel.)

At the default choppiness of 1.0 the naive worst-case error is **725 mm against
an RMS wave height of 535 mm** — larger than the waves themselves, and worst at
the crests, which is exactly where a boat or a swimmer would notice.

### Why not Newton

Newton converges quadratically here, and we even have the Jacobian matrix
already sitting in the normal buffer. It still is not worth it: each Newton
step needs three extra bilinear fetches for the gradient terms, so two Newton
steps cost roughly what five fixed-point steps cost — and four fixed-point
steps are already an order of magnitude finer than the bilinear interpolation
between texels can resolve. Newton would be buying precision the sampling
scheme cannot deliver.

### Sampling wraps, it does not clamp

The FFT surface is exactly periodic with period `patch_length`, so the tile
genuinely tiles and a query anywhere in the world is well defined. Clamping
would invent a flat shelf outside the patch. A test checks that
`height_at(x + L, z)` equals `height_at(x, z)`.

Normals are renormalised after interpolation: bilinear blending of unit vectors
does not preserve length, and a renderer that skipped this would show darkened
bands between texels.

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

---

## ADR-012 — Threading: implementation

### One dispatch path, not two

The built-in pool is plugged in through a `pool_dispatch` adapter with exactly
the public `ParallelForFn` signature, and `run()` calls `dispatch(...)` with no
idea which it is using. There is deliberately no "if we own the threads" branch.
That means the host-scheduler hook is exercised by every test run and every
benchmark, rather than being a lightly-tested side path that breaks the first
time an engine actually uses it.

### Four dispatches, four barriers

```
evolve  ->  FFT rows  ->  FFT columns  ->  finalise
```

The barriers are real data dependencies, not caution: the column pass reads
what the row pass wrote, and finalisation reads what the column pass wrote.
Because each stage is a pure function of the previous stage's **complete**
output, the result cannot depend on how the scheduler interleaved anything.

The FFT row pass is `4 fields x N rows` of independent 1D transforms, flattened
into one index space and chunked once, rather than four separate dispatches
paying four extra barriers.

### Scratch is indexed by task index, not by thread

We cannot know which thread a host scheduler will run a given task on. We do
not need to: two tasks with the same index never run concurrently, so one
scratch block per task index is sufficient and works with any scheduler.

The alternative, `thread_local`, would allocate lazily on first touch — inside
the per-frame path, violating the no-allocation rule the first time a new
thread picked up work.

### The serial threshold, and why it exists

An earlier build threaded unconditionally, and 64² got **slower**: 0.243 ms
serial against 0.296 ms threaded. Four condition-variable round trips cost more
than the work they were distributing. Below 8192 cells the library now runs
serially, through a `serial_dispatch` function so `run()` still needs no branch,
and it does not create a thread pool it would never dispatch to.

8192 is a measured heuristic for this machine, not a law: 64² (4096 cells) lost,
128² (16384 cells) won by 2.3x. The principle is the defensible part —
*threading must never make things worse* — and a threshold is the cheapest way
to guarantee it.

### Verification

Determinism under threading is not assumed, it is tested from three directions:

1. **Bit-identical across thread counts** — 1, 2, 3, 4, 8, 16, 32 workers all
   produce byte-equal buffers.
2. **A reverse-order hook.** A host scheduler that runs indices from `count-1`
   down to `0` still satisfies the contract, and must give identical results.
   This is sharper than a serial hook: it proves there is no hidden dependency
   on task ordering.
3. **A chaotic hook** that scatters indices across raw `std::thread`s in
   whatever order they win the race, repeated across frames.

Plus the pool's own contract: every index runs exactly once, and `parallel_for`
does not return until every slot has been written.

### Measured

512²: **25.6 ms serial → 2.92 ms threaded, 8.76x.** Full table in
BENCHMARKS.md, including why 8.76x rather than 28x (memory-bound workload,
hybrid P/E core CPU, hyperthreading, and four barriers per frame).

---

## ADR-013 — SIMD: runtime dispatch, and bit-exactness as a hard requirement

### One binary, runtime dispatch

The library ships scalar, SSE2, AVX2 and NEON stage kernels and chooses at
startup via CPUID. Promise #1 is "runs on anything", which rules out asking
users to build a separate binary per instruction set.

Only `fft_kernel_avx2.cpp` is compiled with `/arch:AVX2` (or `-mavx2`);
everything else stays at the architectural baseline. Compiling the whole
library with AVX2 enabled would let the compiler emit AVX2 anywhere it liked,
including in the startup path, producing a binary that crashes on older
hardware before it ever reaches the dispatch.

**The OSXSAVE/XGETBV check is not optional.** A CPU can report AVX2 support
while the OS does not save and restore the upper halves of the YMM registers
across a context switch. Using them then silently corrupts state on
preemption — rare, irreproducible garbage. We check XCR0 bits 1 and 2 before
believing the CPUID feature bit.

SSE2 needs no detection (it is part of x86-64) and neither does NEON (mandatory
on AArch64).

### Bit-exactness is part of the kernel contract

Every kernel must match `stage_scalar` **to the last bit**, not merely closely.
That forbids FMA, forbids reassociation, and requires the same operations in
the same order — just several lanes at a time.

The FMA point is the counter-intuitive one: `fmsub` would be *more* accurate
than separate multiply and subtract, because it rounds once instead of twice.
That is exactly why it is banned. If the vector path were more accurate than
the scalar path, the ocean a player sees would depend on which CPU rendered it,
and promise #3 would quietly degrade to "deterministic per machine".

Enforced by a test that `memcmp`s scalar against every kernel in the build, at
every stage and every value of `s`, across sizes from 2 to 1024. It passes on
MSVC with `/fp:precise`, which confirms MSVC does not contract intrinsics into
FMA — something worth verifying rather than assuming.

### Measure before concluding

The first AVX2 build sped the FFT up by only 1.62x, far short of the 8x the
vector width suggests. Timing the row and column passes separately showed the
column pass gaining just 1.27x against the row pass's 1.66x, despite running
identical butterfly code — so the difference had to be memory access, not
arithmetic.

The column pass walks with stride N. Gathering one column touched a separate
cache line per element and used 4 bytes of each 64-byte line.

### The fix: batch 8 adjacent columns

Transforming 8 columns side by side makes each element 8 contiguous floats, so
a fetched cache line is fully used. It also fixes the second problem for free:
batching multiplies the memory stride by 8, so `s >= 8` from the first stage,
and the three early stages that previously fell back to scalar (s = 1, 2, 4,
all below the vector width) now vectorise like the rest.

The elegant part is that **no new kernel was needed**. Batching B interleaved
sequences multiplies every address by B while the twiddles, which depend only
on position within a sequence, stay put. Splitting the *memory* stride from the
*logical* twiddle step was the entire change — one extra parameter — and the
same kernel now serves both the single-sequence row pass and the batched column
pass.

FFT: 11.974 ms to 7.071 ms serial at 512², a further 1.69x. Overall 512²
threaded went 23.620 ms (scalar, single-threaded) to **1.913 ms**, 12.3x.

### What is left, measured rather than guessed

At 512² the serial split is now evolve 5.33 ms (40%), FFT 7.07 ms (53%),
finalise 0.62 ms (5%). Spectrum evolution is still entirely scalar: one `sinf`
and one `cosf` per cell, 262 144 of each per frame.

A vectorised polynomial sine/cosine is the obvious next step, and it carries a
second benefit — a polynomial we evaluate ourselves is bit-identical on every
platform, where libm is not. That would close the last cross-runtime
determinism gap noted in ADR-003.

### The NEON kernel: verified under emulation, not yet on physical silicon

Update (ADR-018): the development machine is x86-64, so there is no physical
ARM board here, but the NEON kernel HAS now been executed — cross-compiled for
aarch64 and run under QEMU user-mode emulation, which translates real ARM64
machine code, including real NEON vector instructions, rather than modelling
the ISA abstractly. The library's own bit-exactness test ran on that build and
passed: the FFT stage kernel matched `stage_scalar` to the bit, with the
harness itself confirming `detect_simd_level() == Neon` so there was no doubt
which code path executed. See ADR-018 for the full methodology and result.

This closes almost all of the original gap. What is left: QEMU's NEON
implementation is a software model of the ISA, and while it aims for (and in
practice achieves) faithful IEEE-754 semantics for the plain multiply/add/
subtract/select operations this kernel uses, it is not a substitute for
running on physical silicon. The specific class of risk that remains is
esoteric ARM floating-point behaviour QEMU might not reproduce exactly - flush-
to-zero handling on denormals is the traditional example, though the kernel
touches no denormal-prone values at the magnitudes an ocean spectrum produces.
The bit-exactness test is cheap to rerun, so the honest recommendation stands:
if this ever runs on a real ARM board (a Raspberry Pi 5, an AWS Graviton
instance, an Apple Silicon Mac under Linux), rerun `ocean_tests` there once and
treat that as the final word - not this paragraph.

---

## ADR-014 — The C API

### The handle is the C++ object

`ocean_sim` is declared but never defined, and the implementation is a
`reinterpret_cast` to `ocean::Ocean`. No wrapper struct, no extra allocation,
no indirection. This is the payoff of ADR-005: because the C++ class is already
a single pimpl pointer, the C API costs nothing on top of it.

### No exception may cross the boundary

Every entry point catches. `std::invalid_argument` becomes a null handle plus
`OCEAN_ERROR_INVALID_ARG`, `std::bad_alloc` becomes `OCEAN_ERROR_OUT_OF_MEMORY`,
and a catch-all covers the rest.

The catch-all is not defensive padding — it is mandatory. Letting an exception
unwind through a C caller's frame is undefined behaviour, and a Rust or C#
caller has no way to catch one regardless.

`ocean_create` returns the handle rather than a status, with the status as an
optional out-parameter. That keeps the common path terse while still letting a
caller that cares distinguish a bad descriptor from an allocation failure.

### `struct_size` is the version tag

`ocean_desc` starts with `size_t struct_size`, which the caller sets to
`sizeof(ocean_desc)`.

Without it, adding one field to the descriptor would make the library read past
the end of an older caller's smaller allocation — garbage at best, a page fault
at worst — and the only fix would be recompiling every binding in the world
against the new header. With it, the library copies exactly the bytes the
caller provided and leaves its own defaults in the rest, so an old binary keeps
working against a new shared library.

A `struct_size` **larger** than we know about is an error, not a truncation:
that caller was built against a newer header, we cannot guess what the extra
fields mean, and silently ignoring a field the caller believes it set would be
worse than refusing.

Tested with a `LegacyDesc` that models the struct as it stood before the
threading fields existed, and requires it to produce byte-identical output to a
current caller using defaults.

### Defaults are read, not duplicated

`ocean_desc_init` fills from a default-constructed `ocean::OceanDesc` rather
than restating the numbers. Duplicated defaults are a classic source of silent
divergence between an API and its wrapper, and a test pins every field.

### Callback types are layout-identical by construction

`ocean_parallel_for_fn` and `ocean::ParallelForFn` are the same signature, so
the translation is a `reinterpret_cast` and a host scheduler pays no trampoline
on the hot path. A test checks that a hook supplied through the C API gives
byte-identical results to the built-in pool.

### Null handles are tolerated everywhere

`ocean_destroy(NULL)` is a no-op like `free()`, and every accessor returns a
sane zero value. Bindings in other languages routinely call into a handle that
failed to construct; crashing inside the library turns a caller-side bug into a
support ticket.

### The header is compiled as real C

`tests/c_header_check.c` is built by the **C** compiler and linked against the
C++ objects. The value is in the build, not the assertions: a "C API" header
that is only ever included from C++ drifts into C++-only constructs within a
release or two, and an `extern "C"` that goes missing shows up as a link error
here rather than in a user's project.

### Verification

The load-bearing test compares the C and C++ APIs with `memcmp` across several
times, plus direct comparison of query results. Anything that changed the
numbers — a float narrowed somewhere, a defaulted field diverging — would mean
bindings quietly simulate a *different ocean* from the C++ API, which is the
kind of discrepancy nobody finds for months.

---

## ADR-015 — The Vulkan viewer

### It is a demo, not a rendering library

The viewer lives in `examples/viewer/` behind `OCEAN_BUILD_VIEWER=OFF`. The
core library still builds with no Vulkan SDK present.

The alternative considered was a reusable `ocean_render` library. It was
rejected because it contradicts the project's own positioning: an engine
already has a render graph, a material system and a swapchain, and is never
going to adopt ours. Shipping a renderer would put Vulkan in the public
headers to serve a user who does not exist. A *demo*, on the other hand, proves
the library works and doubles as the reference integration — which is exactly
what an engine-agnostic library should ship.

The dependency policy is the one `tests/` already established: an optional
target may have dependencies, the library may not.

### The integration is two memcpys

This is the payoff of ADR-004, and it is worth stating plainly because it is
the whole argument for that decision:

```cpp
memcpy(staging,                  b.displacement, N*N*16);
memcpy(staging + texture_bytes,  b.normal,       N*N*16);
// two vkCmdCopyBufferToImage. Done.
```

No repacking pass, no per-component conversion, no quantisation. The textures
are `R32G32B32A32_SFLOAT` because that is literally what the library hands
over. Everything else in `ocean_view.cpp` is Vulkan object creation.

### Tiling is free because the surface is exactly periodic

Seven-by-seven tiles are a single instanced draw; the instance index becomes
the tile offset in the vertex shader. A `REPEAT` sampler is *correct* rather
than approximate here — a wrapped sample at u = 1.05 is genuinely the right
value, because the FFT surface has period `patch_length` exactly. There is no
blending, no overlap and no edge fixup anywhere, and the wireframe capture
confirms the mesh is continuous across tile boundaries.

### Vulkan 1.3: dynamic rendering and synchronization2

Both are core in 1.3 and together they delete a large amount of boilerplate —
no `VkRenderPass`, no `VkFramebuffer`, and the modern barrier API instead of
the legacy one. Every GPU that can run this demo supports 1.3.

Two synchronisation details that are easy to get wrong and only fail on some
drivers:

- **One texture set per frame in flight.** We wait only on the fence for *our*
  frame index, so the other in-flight submission may still be sampling the
  textures when we overwrite them. Double-buffering is cheaper and simpler than
  the barriers a single set would need.
- **One render-finished semaphore per swapchain image, not per frame.** The
  semaphore a present waits on must belong to the image being presented, or a
  fast-recycling swapchain can end up waiting on one that is still pending.

### Shaders are compiled at build time and embedded

`glslc` produces SPIR-V, and a CMake script turns it into a `uint32_t` array in
a generated header. No runtime asset paths — which are the usual reason a demo
fails to launch from a different working directory — and no way to run a shader
that does not match the binary it shipped with.

The sky model lives in a shared `common.glsl` because the water reflects it. If
the sky pass and the reflection disagreed, the horizon would visibly seam where
the ocean meets the sky.

### No back-face culling on the water

A choppy surface genuinely folds over itself at breaking crests. Culled
triangles would punch visible holes exactly where the most interesting geometry
is, so the water draws double-sided and the fragment shader flips normals that
face away from the viewer.

### Verified by looking at it

A `--screenshot` mode renders N frames, copies the swapchain image back and
writes a BMP. The renderer was checked by inspecting those images, not by
assuming that "it compiled and did not crash" meant it worked. Validation
layers report no errors.

What the captures confirm: the sun glitter path is emergent from the wave
slopes rather than faked; Fresnel behaves correctly (transparent near, mirror
at grazing angles); foam appears along crest lines at high wind, driven by the
Jacobian; and there are no tile seams.

### Known limitation

The mesh has no level of detail — every tile is the same resolution, so distant
triangles collapse to sub-pixel size and waste vertex work. Tessellation
shaders with distance-based tessellation levels are the fix, and were rejected
for now because two extra pipeline stages would obscure the integration story
the demo exists to tell.

---

## ADR-016 — Our own sine and cosine, and a vectorised evolve stage

### Two reasons for a hand-written sincos, and the second matters more

**Speed.** The per-frame path needs a sine *and* a cosine of the same angle for
every grid cell — 262 144 pairs per frame at 512². Calling `std::sinf` and
`std::cosf` separately performs the argument reduction twice, and neither can
be inlined into a vector loop.

**Determinism, which is the real reason.** ADR-003 left one gap open and said
so: libm's transcendental functions are permitted to differ by an ULP between
implementations, so a build on glibc and one on MSVC could produce oceans
differing in the last bits. A polynomial we evaluate ourselves is the same
sequence of IEEE multiplies and adds on every platform. **This closes that gap**
— "same seed, same waves" is now true across platforms, not merely across runs
of one binary.

The algorithm is the standard Cephes single-precision reduction: fold into
`[-pi/4, pi/4]` by quadrant with a two-part Cody-Waite subtraction, evaluate a
minimax polynomial, then select and sign by quadrant. Measured error against a
double reference over `[-2pi, 2pi]`: **1.47e-7 for sine, 1.29e-7 for cosine**,
about 1.2 ULP — as good as float can represent.

`floor(v + 0.5)` finds the quadrant, not `std::round` (a libm call) or
`nearbyint` (depends on the current rounding mode, which we do not control).
A fixed rule is what lets the SIMD kernel reproduce it exactly.

### `fmod` was as expensive as the sine

Replacing `std::sinf`/`cosf` with the polynomial gained 1.33x on the stage.
Replacing a single `std::fmod` in the same loop — used to fold `omega*t` into
`[0, 2pi)` — gained another 1.17x on top. `fmod` is a libm call that cannot be
inlined into a vector loop, so it was both slow and an outright blocker to
vectorising the stage at all. `d - floor(d)` on the scaled value replaces it,
and `floor` is one instruction everywhere.

The expectation going in was that the transcendentals dominated. They did not,
quite. Measuring each change separately is what revealed that.

### Bit-exactness had to be designed in, not hoped for

Two details that would each have produced a one-bit mismatch:

**The multiply order.** The scalar reduction first read
`omega * time * kInvTwoPi` — two multiplies with an intermediate rounding —
while the vector path would naturally use one multiply by a precomputed
`time/(2pi)`. Those disagree in the last bit. The scalar function now takes the
hoisted product as its argument, which fixes the operation count and order for
both paths.

**Signed zero.** The scalar code writes `-sx * hr`, which parses as
`(-sx) * hr`. When `sx` is `+0.0` and `hr` is positive, IEEE gives `-0.0`. The
obvious vector spelling `0.0 - (sx*hr)` gives `+0.0` instead, because `x - x`
is `+0` in round-to-nearest. And `sx` **is** exactly zero — at the DC bin,
every frame. The vector kernel negates by flipping the sign bit instead, which
is bit-identical to unary minus for every finite value including negative zero.

Both are enforced by a `memcmp` test across four grid sizes and six times,
including t = 10 hours where the double phase fold is doing real work, and
including partial row ranges because that is how the scheduler calls it.

### AVX2 gave 7.66x here, against 1.62x for the FFT

Same instruction set, same vector width, wildly different results — and the
reason is instructive. The evolve stage streams eight input arrays and eight
output arrays with unit stride, so every cache line fetched is fully used and
the work is genuinely compute-bound. The FFT's column pass was touching one
float per 64-byte line.

This is the whole argument for profiling rather than assuming wider registers
mean proportionally faster code.

### Coverage is deliberately asymmetric

The FFT has scalar, SSE2, AVX2 and NEON kernels. The evolve stage has scalar
and AVX2 only.

That is a measured trade-off, not an oversight. The FFT is the dominant cost at
every size, so it earns full ISA coverage. Evolve is now 5% of frame time, and
an SSE2 version would need a software `floor` (SSE4.1 introduced `roundps`, and
SSE2 is the baseline we promise), while a NEON version could not be tested
here. Both remain straightforward to add if the measurements ever justify it.

### Result

Evolve: **5.334 ms to 0.449 ms serial at 512², 11.9x.** Overall the ocean went
from 23.620 ms single-threaded scalar to **1.426 ms**, a 16.6x improvement, with
the FFT now 84% of what remains. BENCHMARKS.md records the specific remaining
opportunity — the row pass still runs its first three stages scalar, where the
stride is below the vector width.

---

## ADR-017 — Testing low-end hardware without owning it

### The problem

Promise #1 is "runs on anything," but the only machine available for
development is a 2024 28-thread AVX2 laptop. Claiming the library works on
weak hardware without any way to check it would be exactly the kind of
unverified claim this project otherwise avoids.

### The mechanism: force, don't guess

`core/cpu_features.hpp` already had `force_simd_level()` for the SIMD
bit-exactness tests (ADR-013). It is now exposed publicly as
`ocean_force_simd_level()` in the C API, and both `ocean_bench` and the viewer
gained `--simd` / `--threads` flags that call it before constructing anything.

This is a real feature, not just a testing convenience: it is a QA/support
hook. "Does this bug repro on scalar?" is now answerable without different
hardware, for any integrator, in production. `force_simd_level` clamps to what
the CPU can actually execute, so requesting AVX2 on a machine without it falls
back rather than faulting - the same safety property the internal version
always had.

### What this can and cannot tell you

It is a **lower bound** on real slowdown. Dropping SIMD width and thread count
measures the *algorithmic* cost on today's clock speed and cache hierarchy. A
real budget or decade-old CPU is also clocked lower and has a smaller cache,
both of which cost more on top of what this measures. Treat the numbers as
"at least this much slower," not "exactly this much slower."

It also cannot simulate a weak *GPU* for the viewer - only the CPU-side
simulation cost is under this library's control, and the viewer's Vulkan
device selection can be pointed at the integrated GPU with a future flag if
that comparison becomes useful.

### Verifying the render pipeline end to end, not just the buffers

The existing tests prove the *library's output buffers* are bit-identical
across SIMD levels and thread counts. That is necessary but not sufficient for
the promise as stated to users: "the ocean looks the same regardless of your
hardware" is a claim about pixels, not about `float` arrays nobody looks at
directly.

To check that, `--screenshot` was run twice with identical scene settings -
once native (AVX2, 28 threads), once forced to worst case (scalar, 1 thread) -
and the two captures were byte-compared: **zero differing bytes** across a
4.3 MB image. This demonstrates the determinism chain all the way from kernel
dispatch through the render pipeline to the final pixel, which is a stronger
claim than the unit tests alone establish.

### A real bug found along the way

Making that comparison meaningful required fixing `--screenshot` first: it
advanced simulation time using measured wall-clock `dt`, so the same
`--frames N` landed at a different simulated time depending on how fast the
machine rendered the earlier frames. Two screenshots from different
configurations - or even the same configuration on two different machines -
were not comparable by construction, regardless of anything about the ocean
simulation itself. Screenshot mode now advances by a fixed virtual 1/60 s
timestep, decoupled from wall-clock render speed. This is a general
correctness fix for reproducible visual regression testing, independent of the
SIMD-forcing feature that surfaced it.

---

## ADR-018 — Verifying the NEON kernel without owning ARM hardware

### The problem

ADR-008 shipped a NEON kernel for the FFT with an honest caveat: it had never
been executed, because the development machine is x86-64. A kernel that
compiles but has never run is not evidence of anything - the AVX2 kernel's
signed-zero bug (ADR-016) and the FMA-precision trap (ADR-013) were exactly the
kind of thing that only surfaces when the code actually executes, not when it
is merely read.

### The mechanism: cross-compile, then run under emulation - not just compile

Compiling for ARM with the cross toolchain proves the code is syntactically
valid ARM and uses the intrinsics correctly enough to assemble. It proves
nothing about whether the arithmetic is correct, because a compile-only check
never executes a single NEON instruction.

The fix was to actually run it. `aarch64-linux-gnu-g++` (from Ubuntu's
`g++-aarch64-linux-gnu` package, installed into a WSL2 Ubuntu instance since
the host is Windows) cross-compiles a fully static `ocean_tests` binary for
aarch64. `qemu-aarch64-static` (from `qemu-user-static`) then executes that
binary directly: QEMU's user-mode emulation translates real ARM64 machine
code - including real NEON vector instructions - into the host's instructions,
rather than modelling the ISA's behaviour abstractly. This is the same
technique large open-source projects use to cover ARM in CI without owning ARM
hardware.

Two things had to be gotten right for this to work at all, both because CMake
and QEMU each have a sharp edge here:

**`CMAKE_SYSTEM_PROCESSOR` must be set explicitly.** Simply overriding
`CMAKE_CXX_COMPILER` to the cross compiler does *not* make CMake think it is
cross-compiling; `CMAKE_SYSTEM_PROCESSOR` silently stays as the host's
`x86_64`. That would have made this project's own
`CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|...)"` guard (ADR-013) incorrectly
true on an ARM build, and it would have tried to pass `-mavx2` to the ARM
compiler. A proper toolchain file setting `CMAKE_SYSTEM_NAME Linux` and
`CMAKE_SYSTEM_PROCESSOR aarch64` fixes this, and also flips on
`CMAKE_CROSSCOMPILING`, which is what lets `CMAKE_CROSSCOMPILING_EMULATOR
qemu-aarch64-static` make `ctest` transparently run cross-compiled tests
through the emulator.

**The binary must be statically linked.** `qemu-aarch64-static` executes the
ARM machine code itself but still needs an ARM dynamic linker
(`/lib/ld-linux-aarch64.so.1`) and ARM shared libraries to satisfy a
dynamically-linked binary's runtime dependencies - and none of that exists on
an x86_64 host without installing a full ARM sysroot. `-static` removes the
dependency entirely for the one-off purpose of running this test binary.

### A real bug this caught immediately

The very first cross-compile attempt failed - correctly. `tests/test_simd.cpp`
called `evolve_rows_avx2` directly, guarded only by a *runtime* check
(`if (max_simd_level() != Avx2) return;`). But `evolve_rows_avx2` is *declared*
only inside an `#if defined(__x86_64__) || ...` block in `evolve.hpp` (ADR-016
deliberately does not build an AVX2 evolve kernel for other architectures), so
on aarch64 the symbol does not exist at all - a link-time failure, not
something a runtime guard can prevent. The test needed the same compile-time
`#if` around the whole `TEST_CASE`, not just a runtime skip inside it.

This is exactly why "we should test on other platforms" is worth doing rather
than asserting: the bug was in the *test suite's* portability, not the
library, and nothing short of an actual cross-compile would have found it. It
was fixed once, here, and is now permanent - it costs this project nothing on
future x86 builds.

### The result

```
detected SIMD level: NEON
[doctest] test cases:     91 |     91 passed | 0 failed | 0 skipped
[doctest] assertions: 633425 | 633425 passed | 0 failed |
```

(91, not 92: the AVX2-only test correctly does not exist on this target - see
above.) Pulling out just the NEON-relevant assertions:

```
tests/test_simd.cpp:187: SUCCESS: REQUIRE( std::memcmp(a_re.data(), b_re.data(), n * sizeof(float)) == 0 )
  logged: level_name(level) := NEON, n := 4
```

The FFT stage kernel's NEON path was actually exercised - confirmed by the
harness's own `level_name(level)` capture, not inferred - and matched
`stage_scalar` bit for bit, across every size and stage the test suite already
checks for every other kernel.

### What this does and does not prove

It proves the NEON kernel is arithmetically correct on real ARM64 machine
code, not merely that it compiles. It does **not** prove correctness on
physical ARM silicon: QEMU's NEON implementation is a software model, and
while it targets (and in practice achieves) faithful IEEE-754 semantics for
the plain multiply/add/subtract/select operations this kernel uses, it is not
a substitute for real hardware. The specific residual risk is exotic ARM
floating-point behaviour QEMU might not reproduce exactly - flush-to-zero
handling on denormals is the standard example - though the kernel never
touches denormal-range values at the magnitudes an ocean spectrum produces.

### Reproducing this

`scripts/verify-neon-qemu.sh` automates the whole procedure: install the
prerequisites (`g++-aarch64-linux-gnu`, `qemu-user-static`, `cmake`,
`ninja-build`) inside a Linux environment (WSL2 on Windows), then run the
script from the repo root. It cross-compiles and runs the full test suite
under emulation in one step, so this is a repeatable check rather than a
one-off manual procedure - and the natural thing to wire into CI once this
project has any.

If this ever runs on physical ARM hardware (a Raspberry Pi 5, an AWS Graviton
instance, an Apple Silicon Mac under Linux), running `ocean_tests` there once
is the remaining, final word - not this ADR.

---

## ADR-019 - Shallow-water dispersion, and a general rule for C struct versioning

### The physics: one function changes, nothing else does

Deep water assumes depth exceeds roughly half the longest represented
wavelength, which is true for open ocean but false near a coastline. The full
relation is `omega^2 = g k tanh(k h)`, deep water being its `h -> infinity`
limit. Everything else in the spectrum - the JONSWAP frequency-space shape,
the directional spreading, fetch and wind - is defined in frequency space and
is unaffected by depth; only the mapping between frequency and wavenumber
changes. That mapping is used in exactly two places: `wave_density()`'s
`domega/dk` Jacobian, and the per-cell `omega` table `build_spectrum()` writes
once at construction. Both now call a shared `dispersion_omega` /
`dispersion_domega_dk` pair instead of a hardcoded `sqrt(g*k)`.

This is deliberately the *dispersion-relation* generalisation only, not the
*spectral-shape* generalisation. A more complete treatment (the TMA spectrum)
also reshapes the JONSWAP curve itself for shallow water; that is a further
refinement this does not attempt, and `SpectrumDesc::depth`'s doc comment says
so rather than implying this is the full physical picture.

**The deep-water branch is a literal special case, not a numerical
coincidence.** `depth <= 0` returns `sqrt(g*k)` directly rather than evaluating
`tanh(k*depth)` for some very large depth that happens to be close to 1. Every
existing deep-water test, and every existing deep-water ocean, is therefore
provably unaffected by this feature's existence - confirmed by the build
having identical assertion counts before and after this change landed.

**`sech^2(kh)` is computed as `1 - tanh(kh)^2`, not `1/cosh(kh)^2`.** `cosh`
overflows a double for `kh` above a few hundred, while `tanh` saturates safely
to 1.0. A coastal depth combined with a short-wave wavenumber routinely pushes
`kh` into the thousands, so this is a real, not theoretical, overflow risk.

### Verification

Beyond the usual bit-level checks, this adds tests anchored to closed-form
textbook results, in the style of the fetch-scaling law in ADR-009:

- **The very-shallow limit recovers `omega -> k sqrt(g h)`** - the classic
  non-dispersive shallow-water wave speed (why a tsunami, an extremely long
  wave over ordinary depth, is non-dispersive and moves at a speed set by
  depth alone).
- **`domega/dk` is checked against a central finite difference** across deep,
  shallow and transitional depths - this validates the calculus directly,
  independent of any physical interpretation, which is what catches an algebra
  slip (a sign, a missing factor of 2) in a hand-differentiated formula.
- **A monotonicity property**: `tanh(kh) <= 1` always, so shallow-water omega
  can only be less than or equal to deep-water omega at the same k, never
  greater. A sign error in the tanh argument would violate this immediately.
- **The polar energy-conservation identity (ADR-009's strongest check)**,
  re-run at a genuinely shallow depth. This is what catches
  `dispersion_omega` and `dispersion_domega_dk` disagreeing with *each other*
  even if each passed its own unit tests above.

Measured, not asserted, at 12 m/s wind: switching from deep water to a 4 m
depth raises realised spectral energy by about 15% and visibly shortens and
compresses the wave crests in the viewer (screenshots taken with the same
`--frames`/seed for a fair comparison) - consistent with real shoaling
(Green's law: energy density rises as group velocity falls), though no
specific magnitude is asserted in the test suite, since deriving the exact
expected number would require solving the flux-conservation integral this
project has not attempted; the test only requires the two differ.

### A general finding about C struct versioning: fields can hide in padding

Adding the C API's `water_depth` field surfaced something worth documenting on
its own, independent of ocean physics: `ocean_desc` grew a field with **no
change in `sizeof()`**. `thread_count` (the previous last field, a `uint32_t`)
ended 4 bytes short of the struct's required 8-byte alignment - forced by the
`size_t`/pointer/`uint64_t` members elsewhere in the struct - so the compiler
was already inserting 4 bytes of invisible tail padding. `water_depth` (also
4 bytes) exactly fills that padding rather than growing the struct further.

This matters for ADR-014's `struct_size` mechanism generally: **a caller's
`struct_size` cannot always distinguish "built before this field existed" from
"built after," because the two can produce numerically identical struct
sizes.** What still makes the old caller safe is the *other* half of the
convention this API already documented: `ocean_desc_init()` zero-fills the
entire struct before setting known fields, so a caller who follows that
convention has zeros - not garbage - sitting in what was their padding, and
zero happens to be `water_depth`'s correct default (deep water). A caller who
hand-rolled their struct without zeroing it first would not be protected by
`struct_size` alone in this specific case.

The practical rule this yields, for every future field: **appending at the
true end of the outermost struct keeps `struct_size` meaningful; relying on
`struct_size` changing at all is not guaranteed; and `ocean_desc_init` is not
a convenience, it is the load-bearing half of the compatibility promise.**
Verified with two chained legacy-struct tests: one from before the threading
fields existed (genuinely smaller, `struct_size` differs), and one from
immediately before `water_depth` (numerically the same size, protected only by
zero-initialisation) - the second is the one that actually exercises this
finding rather than the easier case.

---

## ADR-020 - Cascades: multiple ocean patches summed into one sea state

### The problem

A single Ocean patch must compromise between two conflicting needs: a large
`patch_length` to capture big swells, and many texels per metre to capture
fine ripples. Push one and the other suffers. Cascades resolve this the way
every shipping ocean does: run several small, independently-scaled patches
side by side, each covering its own wavelength band, and sum their
contributions.

### API shape: `CascadeStack` wraps N `Ocean` instances

```cpp
using CascadeLevel = OceanDesc;   // a cascade level IS an ocean patch

class CascadeStack {
public:
    explicit CascadeStack(std::span<const CascadeLevel> levels);
    void update(double time);
    Buffers buffers(std::size_t level) const noexcept;   // per level, NOT combined
    float height_at(float x, float z) const noexcept;    // combined
    Surface sample_at(float x, float z) const noexcept;  // combined
};
```

`ocean::Ocean` itself is completely unchanged - every existing single-patch
integration is unaffected by this addition, and `CascadeStack` is implemented
entirely in terms of the existing public `Ocean` API, adding no new access to
internals.

**A real correction made mid-design, worth stating plainly:** the initial plan
described `CascadeStack::buffers()` as returning "one set of combined
buffers." That was wrong, and the mistake is instructive. Each level keeps its
own resolution and `patch_length` - that is the entire point of cascades - so
their texel grids do not share a common indexing and cannot be flattened into
a single CPU array. `buffers(level)` therefore returns each level's own
buffers unmodified; a renderer samples each one at world position mapped into
that level's own UV space and sums the *samples*, not the raw arrays. This is
exactly what "additive sum, sampled at the same world position, each in its
own patch's space" already implied - the buffer-representation question was
underspecified in the plan, not a new decision.

### What is exact, and what is approximate

**Height and horizontal offset are exact linear superposition.** Each level's
height/offset field is an independent real-valued function of world position
(its own periodic tiling). Summing real fields at the same point is not an
approximation - it is what "the composite surface is the sum of several wave
trains" means mathematically, and it is the same principle Tessendorf's
original formulation already relies on for summing individual *wavenumbers*
within one spectrum, just applied across cascades instead of within one.

**Normal and foam are a stated approximation.** The public `Ocean` API
returns already-normalised normals and an already-thresholded foam scalar
(ADR-005's pimpl deliberately does not expose raw slope/Jacobian gradients).
`CascadeStack::sample_at` therefore combines normals by summing each level's
unit normal and renormalising, and combines foam via
`1 - product(1 - foam_i)` - a "screen"/OR blend, standard for combining
independent [0,1] coverage layers. Both are the well-understood techniques
real-time engines already use for multi-scale detail-normal blending; neither
is a re-derivation from first principles, and both are documented as such
rather than presented as exact.

**The one approximation that also touches height, not just normal/foam:**
each level solves its OWN fixed-point inversion (ADR-011) against only its
own chop, not the combined chop of every level. The height/offset SUM is
exact given each level's own answer; what is approximate is that a level's
own answer assumes it is the only source of horizontal displacement, when in
a multi-cascade scene it is not. A fully joint solve across all levels'
combined displacement would remove this residual, and is the natural
refinement if cascade choppiness is ever pushed high enough for the cross
term to matter visibly - untested here because it has not yet visibly
mattered at any setting exercised.

### Verification: variances of independent levels add

The strongest check, in the spirit of ADR-009's fetch-scaling law and
ADR-010's Parseval identity: if three cascades are statistically independent
(distinct seeds) and spectrally non-overlapping (`small_wave_cutoff` and
`patch_length` chosen so their wavelength bands do not overlap much), then for
the sum `h = h1 + h2 + h3`,

```
Var(h) = Var(h1) + Var(h2) + Var(h3)
```

exactly, because `Var(a+b) = Var(a) + Var(b) + 2*Cov(a,b)` and `Cov -> 0` for
independent fields. Measured across a dense, deliberately non-grid-aligned
set of world-space samples: composite variance matched the sum of the
individual levels' variances to within 8% (the residual being Monte Carlo
sampling noise plus the small, expected spectral overlap at each cascade's
soft-knee cutoff edge - see ADR-009 on why the cutoff is a knee, not a wall).
This is a strictly stronger claim than "the composite differs from any one
level," and it is a real, derivable statistical property, not a
plausibility check.

Also verified: a single-level `CascadeStack` is indistinguishable from a
standalone `Ocean` (both buffers and every query, bit-for-bit where exact and
to float precision where the query path's fixed-point iteration is involved);
determinism (bit-identical output across repeated construction, seeking
backwards reproduces exactly); and `update()` performs no heap allocation
beyond construction, preserving `Ocean`'s own guarantee transitively.

### Performance: linear, as expected

Measured on the i7-14700HX, 256x256 per level: a single `Ocean::update()` is
0.512 ms (median); a 3-level `CascadeStack::update()` is 1.549 ms - a 3.02x
ratio, matching the fact that `update()` runs levels sequentially (each
already parallelises internally; see below for why sequential was chosen).
`height_at()` costs 0.4 microseconds - three independent fixed-point solves,
each cheap.

**Why sequential across levels, not interleaved.** Each level already
parallelises across its own hardware threads (or the shared host scheduler).
Interleaving levels' internal stages - running level 0's FFT concurrently
with level 1's spectrum evolution - is a plausible future optimisation, but
only worth the added complexity once profiling shows the sequential form
actually limits a real scene, consistent with this project's standing
practice of measuring before optimising rather than anticipating a cost that
has not been shown to matter.

### The viewer: a real bug caught by tracing through what --cascades 1 would sample

The viewer demonstrates exactly 3 cascades (far/mid/near), fixed at that
count in `OceanView` to keep the descriptor layout and shader loops simple -
`kMaxCascades = 3` is a demo-only simplification; `CascadeStack` itself is not
limited to 3.

Extending the vertex/fragment shaders to sum 3 cascades' worth of
displacement/normal textures needed a descriptor layout with 7 bindings (1
UBO + 3 x 2 texture pairs). For `--cascades 1` or `2`, the unused slots
initially fell back to aliasing level 0's real texture, on the reasoning that
"the shader always reads a valid, bound image." Tracing through what that
actually computes catches the bug immediately: with slots 1 and 2 both
pointing at level 0, the vertex shader's `d0 + d1 + d2` sum counts level 0's
displacement three times instead of once, and `1 - product(1-foam_i)` triple
counts its foam contribution - silently changing the rendered sea state
whenever fewer than 3 cascades are active, with no validation error to catch
it (every descriptor is legitimately bound to *something*).

The fix is a dedicated 1x1 dummy texture pair (displacement all-zero, normal
+Y unit, uploaded once and never touched again) that unused slots point at
instead: it contributes exactly nothing to height/offset and a neutral
normal to the sum, correctly modelling "this cascade does not exist" rather
than "this cascade exists twice." This was caught by mentally executing the
shader's arithmetic for the `--cascades 1` case before ever running it - the
same habit of tracing through what code actually computes, rather than
trusting that "it will bind to something" is enough, that the earlier
signed-zero bug in the AVX2 evolve kernel (ADR-016) and the FMA-precision
trap in the SIMD FFT kernels (ADR-013) were also caught by.

### A debugging note worth recording: a false alarm, and how it was resolved

The first cascade renders appeared to show a washed-out, overexposed sky
compared to earlier pre-cascade screenshots. Chasing it methodically - not by
guessing, but by falsifying hypotheses in order - eventually showed there was
no regression: raw pixel sampling of the "washed out" region matched a known-
good pre-cascade reference image byte-for-byte at every sampled coordinate.
The perceived difference was real but not a bug: the far cascade's own
`small_wave_cutoff = 15 m` deliberately removes fine ripple detail (that is
cascade 2's job when all three run together), so viewed in isolation
(`--cascades 1`) its smooth, low-contrast surface reads as hazier than a
normal single-patch ocean, without a single incorrect pixel anywhere.

The methodology is worth recording because it generalises: rendering bugs
that "look wrong" are not always code bugs, and the fix for uncertainty is
the same either way - hardcode a shader stage's output to a known constant to
prove the pass executes over the right region; visualise intermediate
quantities (a view ray, a light direction, a dot product) as raw colour to
check they are not NaN or degenerate; and, decisively, diff actual pixel
values against a trusted reference rather than trusting a rescaled,
recompressed visual impression. The last of these settled the question in one
step after several plausible-sounding hypotheses had already been ruled out.

---

## ADR-021 - Local interaction: iWave on top of the FFT surface

### The physics, and why the operator is nonlocal

Linearised, irrotational, inviscid free-surface flow. Laplace in the volume,
kinematic and dynamic boundary conditions linearised onto `y = 0`:

```
dh/dt = dphi/dy |_0 ,      dphi/dt = -g h |_0
```

Differentiating the first in time and substituting the second gives

```
d^2h/dt^2 = -g * G{h},     symbol of G = |k|
```

because the decaying harmonic extension of a mode `e^{ik.x}` into the
half-space is `e^{|k|y}`, so `d/dy` at the surface is multiplication by `|k|`.
Substituting a plane wave recovers `omega^2 = g|k|` - **the same deep-water
dispersion relation the FFT ocean already solves** (ADR-009). That shared
origin is what makes adding the two fields legitimate rather than a blend, and
it is the same argument ADR-020 makes for cascades, applied across a different
axis.

`|k|` is not a polynomial in `k`, so `G` is not a differential operator. It is
`sqrt(-laplacian)`, and in real space it is a convolution with infinite
support. The iWave idea is to truncate that convolution to a small radius `P`
and apply it directly, which keeps the operator **local**.

### Why locality is the whole product

An FFT convolution on the local grid would be exact (no truncation) and, at
256 squared, actually *cheaper* - roughly 80 flops per cell against 169 taps.
It was rejected, and this is the argument an interviewer is most likely to
press on:

- FFT convolution is **cyclic**. A ripple leaving the right edge re-enters on
  the left. Zero-padding to 512 squared to prevent that costs 4x and erases the
  advantage.
- The **absorbing layer** is a spatially varying damping. There is no such
  thing in a global spectral multiply.
- The **obstruction mask** is a spatially varying boundary condition.
  Enforcing it spectrally needs an iterative solve per substep.

Direct convolution buys locality, and locality is what makes a local field
worth having at all.

Against the other obvious alternative - a plain wave equation
`d^2h/dt^2 = c^2 laplacian h` - the answer is dispersion. That scheme is 9
flops per cell instead of 169, but every wavelength travels at the same `c`, so
an impact gives one rigid expanding ring. Real water gives a ring that fans
into a train, long waves running ahead of short ones, and that dispersion *is*
the look. The honest framing is that **iWave IS the wave equation with the
dispersion correction taken to its limit**: the correction is exact, and the
price is that the exact spatial operator is nonlocal. The kernel is not a hack
bolted onto a wave equation; it is the correct linearised operator, truncated.

### The kernel: fitted to a band, not truncated from a transform

Two derivations are implemented so the choice is a measured number.

**Hankel** is the published derivation: the inverse Hankel transform of the
symbol (radial symmetry collapses the 2-D transform to
`(1/2pi) integral S(k) J0(kr) k dk`), evaluated to the grid Nyquist and
truncated at radius `P`. Kept as the reference and cross-check. `J0` is written
out from Abramowitz and Stegun 9.4.1 / 9.4.3 because `std::cyl_bessel_j` is a
C++17 special-math function MSVC does not implement, and a dependency is not an
option (ADR-001).

**LeastSquares**, which ships, solves directly for the truncated stencil whose
*realised* symbol best matches the true one over the band the field claims.
Measured peak symbol error over wavelengths of 2 to 16 cells, deep water,
0.25 m cells:

| P  | taps | Hankel | LeastSq | LSQ wave-speed error |
|---:|-----:|-------:|--------:|---------------------:|
|  3 |   49 | 63.4%  |   36.1% |                18.0% |
|  4 |   81 | 61.1%  |   26.5% |                13.3% |
|  6 |  169 | 59.7%  |   13.1% |                 6.6% |
|  8 |  289 | 59.6%  |    6.5% |                 3.2% |
| 10 |  441 | 59.7%  |    3.0% |                 1.5% |

Speed error is half the symbol error, because `omega = sqrt(g S)`.

Two constraints are enforced exactly rather than approximately. **D4 symmetry**:
only the octant `a >= b >= 0` is stored and every orbit member is written from
one value, so the 8-fold symmetry is bit-exact rather than
equal-after-rounding. **Zero sum**: the true symbol is exactly zero at `k = 0`,
so the centre tap is *eliminated* algebraically (`g00 = -sum n_ab g_ab`) rather
than penalised, making `Shat(0) = 0` hold to the last bit. Without it a uniform
water level would feel a restoring force and oscillate in place forever,
because the operator cannot propagate DC anywhere.

### Three findings from building the kernel

**The Hankel transform needs a cell-area factor.** It returns the *continuum*
kernel, a density in 1/m^2; the discrete stencil approximates the convolution
integral by a sum over cells, so each tap carries `dx^2`. The first build was
wrong by exactly `1/dx^2` - and being a constant, it survived every increase of
`P`, so it looked nothing like truncation error. The clue was that the error
barely moved between P = 2 and P = 10.

**The truncated-transform kernel has a latent instability, and it is not in the
band anyone looks at.** Its symbol is faithful throughout the inscribed Nyquist
disc but goes **negative in the corners of k-space** - reaching -0.12 to -0.18
of its maximum, at every radius from 2 to 12. Those corner modes are shorter
than two cells along the diagonal, so they are aliasing artefacts rather than
waves; but the grid still holds them, and `omega^2 = g S < 0` means they grow
exponentially instead of oscillating. Nothing deliberately excites a
checkerboard, which is exactly what makes it dangerous: rounding noise seeds
it, damping cannot remove an exponential, and it surfaces much later as "the
water sometimes explodes". Fitting over a band removes it for free, because the
fit includes the corners at low weight - enough to hold the symbol non-negative
there without spending accuracy in the band that matters. `kernel_is_stable()`
checks it and the solver constructor refuses a kernel that fails.

**The first least-squares attempt measured worse than the published kernel**,
and the reason generalises. Weighting by `1/S^2` over the whole plane puts the
*highest* weight near DC - exactly where a finite even stencil can never match
`|k|`, because its symbol is analytic and even and so behaves like `c k^2`
against a true `k`. The fit spent its freedom on the region it could not repair
and paid for it in the region it could. Confining the fit to a band inverted
the result.

That low-k mismatch is a real limit of the method, not a defect of this
implementation, and it is recorded rather than hidden: at P = 6 and 0.25 m
cells, dispersion is within 10% only for wavelengths below 4.05 m. **Long waves
are the FFT ocean's job; short ripples are this field's.** The division of
labour is principled, not a workaround.

### The scheme, and why stable is not accurate

```
h^{n+1} = [ 2h^n - (1 - a dt) h^{n-1} - g dt^2 G{h^n} ] / (1 + a dt)
```

Second order in time; the damping term is centred, so it is unconditionally
dissipative for `a >= 0` and can never add energy whatever the timestep.

Leapfrog on `h'' = -omega^2 h` is stable iff `omega dt <= 2`. The limit is
derived from the kernel's **realised** symbol, not from theory - truncation
ringing can push the realised symbol above the ideal `|k|`, and the bound
depends on the largest frequency the state can actually contain. Computed from
what was built, not from what was intended.

But the discrete frequency obeys `sin(w~ dt/2) = w dt/2`, so at the stability
limit `w~ dt = pi` against a true 2 - a **57% frequency error**. Stable, and
completely wrong. So the default timestep comes from accuracy, not stability:
1/60 s against a 0.1539 s limit, 9.2x of headroom, giving a **0.197% leapfrog
phase error at Nyquist**. 1/60 s is also exactly one substep per frame at
60 Hz, which keeps the common case cheap. A `fixed_dt` above the limit is
refused at construction rather than silently clamped, because a caller who
believes they are running at a timestep they are not has a very hard bug to
find.

A caller cannot push past it by feeding a huge `dt`: an accumulator runs whole
fixed steps, and `max_substeps` caps them and **discards** the backlog rather
than carrying it. After a long hitch the field runs slow; the frame does not
explode. Determinism survives, because the clamp is a function of the
accumulator, which is a function of the dt sequence.

### The impulse shape

The 2-D Ricker wavelet, `(1 - u) e^{-u}` with `u = r^2 / 2 sigma^2`. Three
reasons, in increasing order of how much they matter.

1. It looks like an impact crater: a central depression ringed by a raised rim
   of exactly `e^-2` = 13.5% the depth at `r = 2 sigma` - the shallow broad rim
   real craters have. A Gaussian spike looks like nothing in nature.
2. **Its integral over the plane is exactly zero.** The operator symbol is zero
   at `k = 0`, so any net volume injected can *never propagate away* - it would
   sit there as a permanent bump forever. A Gaussian dimple does exactly that.
   Zero net volume is physically forced, and it is the same statement as "the
   rock displaces water, and the rim IS that displaced water".
3. Its transform peaks at `k = sqrt(2)/sigma`, so `radius` is a **wavelength
   selector**: the dominant emitted wavelength is about `4.44 sigma`. That is
   what makes the dispersion test clean - inject at a known sigma, predict the
   group velocity, measure the ring.

A clipped, cell-sampled Ricker does *not* integrate to zero even though the
continuous one does: the tail beyond 3 sigma carries 5% of the peak. It is now
clipped at 4 sigma with the discrete mean subtracted, exactly as the kernel own
DC term is.

Displacement is added to **both** time levels and velocity to only the older
one, because `dh/dt = (h^n - h^{n-1})/dt`: adding to `h^n` alone would inject a
velocity of `displacement/dt`, an enormous one. `strength` says the water *has*
been pushed down; `velocity_y` says it is *still* being pushed down; a real
impact is both.

### Verification: the ring expansion, decomposed

The test that proves the method is physical measures a real group velocity out
of a real simulation. It compares against **two** references, because they are
two different claims:

| sigma | lambda0 | measured | vs realised | vs ideal |
|------:|--------:|---------:|------------:|---------:|
|  0.30 |  1.33 m |   0.6941 |       0.10% |    0.62% |
|  0.40 |  1.78 m |   0.8042 |       2.47% |    0.97% |
|  0.50 |  2.22 m |   0.9238 |       4.26% |    3.74% |
|  0.60 |  2.67 m |   1.0721 |       1.94% |    9.90% |
|  0.80 |  3.55 m |   1.4363 |       0.71% |   27.51% |

*vs realised* is the solver reproducing the operator it was given - a failure
there is a bug. *vs ideal* is that operator being good deep-water physics - a
gap there is the kernel band limit. At sigma = 0.8 the solver tracks its own
operator to **0.71%** while the ideal error is 27.5%, which localises the entire
gap to the kernel rather than the integrator. Conflating the two would have
made the method look broken when it is not.

The reference is the **energy-weighted** mean group velocity, not `c_g` at the
profile peak wavenumber: the transform goes like `k^2 exp(-s^2 k^2/2)`, so the
radial energy density goes like `k^5 exp(-s^2 k^2)`, a different and broader
peak. Comparing against the wrong number and then congratulating the code for
the mismatch is an easy trap.

**Group velocity is a DERIVATIVE of the symbol**, which is why the ideal-theory
error grows so much faster than the symbol error. Where the realised symbol is
rising steeply through the transition band, a 28% symbol error becomes a 50%
*speed* error. The accurate band is therefore narrower for group velocity than
the symbol alone suggests - a point worth stating before someone else does.

### Boundaries

**Absorbing layer.** Damping ramps up quadratically from the inner edge rather
than switching on at it: a step change in damping is an impedance
discontinuity and reflects almost as badly as the hard boundary it replaces.
Measured: **0.0025%** of launched energy returns to the central region after a
full round trip, and removing the layer makes reflection **1991x** worse - so
the threshold is measuring the layer, not the damping.

**Obstruction.** This is a real physics distinction, not a style option. A
rigid hull is no-normal-flow, `dphi/dn = 0`, which for surface elevation is
`dEta/dn = 0` - a Neumann condition, and a crest reflects as a **crest**.
Zeroing the field at solid cells, which is what the published iWave does and
what nearly every implementation copies, is Dirichlet `eta = 0`: physically a
pressure-release surface, not a wall, and it **flips the sign** of the
reflection. Measured peak signed field in front of a wall: Neumann -0.518,
Dirichlet +0.412. Neumann is the default, implemented by filling each solid
cell from its nearest fluid cell (a multi-source BFS built once per
`set_obstruction`, never per frame); Dirichlet remains available for anyone
comparing against a reference implementation.

### Recentring without a jolt

The shift is always an **integer number of cells**, so it is an exact move:
every retained cell keeps its bit-exact value, nothing is resampled,
interpolated or filtered, and there is no jolt *by construction* - not "a small
jolt we smooth over". Verified by comparing 61254 interior cells before and
after a (10, 7) cell shift: **0 differing bits**. Newly exposed cells are set
to zero, which is the physically correct value since this field holds only the
local disturbance, and they arrive at the grid edge where the absorbing layer
has already attenuated everything to nothing.

Three things that are easy to get wrong: both time levels must shift together
(shifting one misaligns them and turns the second time derivative into noise);
the shift must happen at a substep boundary; and the row iteration order must
follow the shift direction so overlapping ranges cannot corrupt.

### Composition, and why this is a separate object

`Ocean::update()` takes **absolute** time and is a pure function of
`(seed, desc, time)`. That purity is core promise #3 and it is what makes
seeking, pausing and replay work. An interaction field is a time-stepped ODE -
inherently incremental, and it cannot be made a pure function of absolute time.
Folding it into `Ocean` would quietly demote that promise for everyone. So they
stay separate, and the signatures differ so the distinction is visible at every
call site: `ocean.update(absolute_time)` against `field.update(dt)`.

`WaterSurface` composes them for queries, holding non-owning references.
Normals compose through **slopes**:

```
sx = -nx/ny + dEta/dx ,   sz = -nz/ny + dEta/dz ,   n' = normalize(-sx, 1, -sz)
```

This is **exact**, unlike ADR-020 cascade blend. Heights adding in world space
means world-space slopes add - an identity. The cascade sum has to blend
already-normalised unit vectors because the public API exposes nothing else
(ADR-005); here there is a genuine height field, so the exact form is available
and there is no reason to approximate. The fragment shader runs the same
arithmetic, so shading and physics agree rather than each being separately
plausible.

### The viewer, and three bugs found by looking at the output

Left click casts a ray and marches it against the **displaced** surface.
`f(t) = ray(t).y - surface_height(ray(t).xz)` starts positive; find the first
sign change, then refine with **Illinois** (modified regula falsi), which keeps
the bracket - so it cannot diverge the way a secant can on a bumpy surface -
while converging superlinearly rather than one bit per iteration as bisection
does. The march step grows with distance, because perspective means a step one
pixel wide near the camera is many pixels wide at the horizon.

Failure cases, all real: a grazing ray covers a huge horizontal distance per
unit of vertical drop and can step clean over a crest and come down the far
side; a choppy surface genuinely folds at breaking crests, where the height is
multi-valued and "the first crossing" is only approximately meaningful (the
same limit ADR-011 inversion has, inherited); and at long range float precision
becomes comparable to the step. Rays below a slope threshold are refused and
the march is capped at 2 km - a click that does nothing beats a rock 4 km away.
A max-mipmap with cone stepping is the scalable answer and is deliberately not
built for a demo driven by mouse clicks.

Three bugs, all caught by inspecting output rather than trusting that binding a
texture meant seeing it:

1. **The fragment shader used the interpolated per-vertex sample.** The mesh is
   about 3 m per quad while the field carries 0.5 to 4 m ripples, so
   interpolating across a quad aliased them away completely: field present,
   texture bound, water still flat. Sampling the texture per pixel is what
   makes the ripples catch the light rather than merely move the mesh.
2. **The field was recentred on the camera**, not on what the camera was
   looking at. With the default camera the 64 m field covered water behind the
   viewer while the water being looked at fell outside the grid, so every
   impulse aimed at it was silently swallowed.
3. **`recenter()` left the public buffer stale**, so a query between a recentre
   and the next update read the old field through the new origin - every value
   off by the shift, silently.

The interaction texture samples `CLAMP_TO_BORDER` with a transparent-black
border, through its own sampler. This is the exact opposite of the cascade
textures, which `REPEAT` because they are genuinely periodic - and reusing the
repeating sampler that is already to hand would tile one splash across the
entire ocean.

### Performance

Full tables in BENCHMARKS.md. At the default 256 squared, the field costs
**0.191 ms** threaded - about 37% of one `Ocean::update()` and 13% of a 3-level
cascade frame.

The benchmark caught a bug rather than merely reporting a speed, for the second
time in this project (the signed-zero AVX2 bug in ADR-016 was the first).
256 squared measured four times slower *per cell* than 512 squared, which is
impossible for a loop whose work is proportional to cell count. The cause was
**denormal stalls**: the absorbing layer multiplies its cells by about 0.8 per
substep, so after a few hundred steps a ring of cells sits in the denormal
range where x86 arithmetic traps to microcode at roughly 100x cost. Each grid
size had run a different number of substeps by the time it was measured.
Flushing values below 1e-30 to zero gave **6.8x at 256 squared** and 6.2x at
512 squared.

FTZ/DAZ in MXCSR was rejected as the fix: it is a process-wide CPU mode that
would change results in the host own code, and it is an x86 register with no
portable equivalent, so the four kernels would stop agreeing bit for bit and
the ADR-013 contract would quietly break. An explicit compare-and-zero is
portable, deterministic, identical on every path, and covered by the same
bit-exactness test as everything else.

---

## ADR-022 - Persistent foam, and four rendering changes that earn their cost

Four pieces, ordered by impact per unit of effort. Three are viewer-only; the
first is a library feature, because foam that persists is simulation state
rather than a shading trick.

### 1. Foam that persists and drifts (library)

The FFT surface already reported foam, but only as an instantaneous Jacobian
threshold: it appeared the moment a crest folded and vanished the moment it
unfolded, pinned to the wave rather than to the water. Real foam is a passive
tracer - created by breaking, carried by the surface current, dissolved over
seconds.

`FoamField` adds semi-Lagrangian advection by the horizontal orbital velocity
plus exponential decay, using the FFT's own Jacobian foam as the source so the
two agree about *where* foam is born and only persistence and drift are new.
It is a separate object for exactly ADR-021's reason: `Ocean::update()` is a
pure function of `(seed, desc, time)` and advected foam is history-dependent
state that cannot be. It refuses an ocean without `compute_velocity` rather
than silently degrading - without velocity there is nothing to advect with.

Semi-Lagrangian is unconditionally stable, so there is no CFL limit anywhere in
the class; the price is numerical diffusion, which for foam reads as a patch
softening at its edges and is, if anything, welcome.

**Each behaviour is verified in isolation, by switching the other off.**
Decay: predicted `exp(-decay*t) = 0.084882` against measured `0.084883`, a
0.000% error. Advection: a drift of exactly one cell per step makes the
bilinear weights exactly 1 and 0, so advection becomes a pure relabelling and
the result must be **bit-identical** to the field shifted by hand - 16384 cells
compared, 0 differing, and a full lap around the periodic patch returns it
exactly. Testing the two together would only have shown that something changed,
which is the weakest possible claim.

Ping-pong buffers are load-bearing for determinism, not just tidiness:
advecting in place would let one row read what another had already overwritten,
and *which* rows depended on how the scheduler split them.

`foam_at()` inverts the choppy displacement like `Ocean::sample_at` does. Worst
error 0.0007 against 0.9994 for a naive lookup - essentially foam versus no
foam, because foam lives exactly where the displacement is largest.

### 2. A real sky (Preetham)

Most of what you see looking at an ocean is reflected sky, so the reflection is
only ever as good as the thing reflected. A two-colour gradient gives flat
water no matter what the simulation does.

Preetham, Shirley & Smits 1999: Perez's five-parameter distribution fitted to a
spectral atmospheric simulation, driven by one turbidity number. Chosen over
Hosek-Wilkie, which is better near sunset but needs a large coefficient table
baked into the binary - ADR-001's zero-dependency rule makes a compact closed
form worth more than the last few per cent here.

**The bug worth recording is that the model was right and the tonemapper was
eating it.** Per-channel Reinhard compresses a bright channel harder than a dim
one, so it desaturates everything bright toward grey. Measured: the model
produced a zenith of linear `(0.139, 0.236, 0.471)` - blue at 3.4x red, a
proper sky - and the tonemapper delivered `(104, 126, 158)`, nearly neutral. A
sunset rendered grey-blue not because Preetham failed to produce warmth but
because the last line of the shader clipped it away. Tonemapping *luminance*
and carrying chroma through unchanged fixes it, fading back toward per-channel
as luminance climbs so that genuinely intense sources still read as white - a
photograph of the sun is a white disc, not a saturated orange one.

This was diagnosed by porting the shader to Python and evaluating it
numerically, which is what separated "the model is wrong" from "the model is
right and something downstream is destroying it". Guessing from pixels had
already produced two wrong conclusions.

The water's subsurface colour is now tinted by the sky sampled straight up,
normalised against its own luminance so it changes hue without changing
brightness. Light leaving the water is light that entered it; without this a
warm sunset sat over cold cyan water and the image did not cohere.

### 3. Distance-based detail fade

A cascade whose texels project to less than a pixel cannot be resolved. What
reaches the screen is not detail but aliasing, and the worst kind - it shimmers
as the camera moves, because which sub-texel each pixel lands on changes every
frame.

Each cascade now fades once its texel drops below roughly a pixel. This is
mip-mapping's argument applied to a whole frequency band: the cascades *are* a
frequency decomposition, so dropping the top band is what a low-pass filter
would do.

**Measured, not eyeballed.** RMS discrete Laplacian, which isolates pure
high-frequency content, over a band just below the horizon: 24.305 with the
fade off, 9.086 with it on - **62.6% less high-frequency energy**. Over a
near-field band, where detail must be preserved: 14.084 against 12.986, a 7.8%
change, most of which is the normal-combination change below rather than the
fade.

The fade forced a genuine improvement. **A unit normal cannot be scaled** -
multiplying it by 0.5 does not halve the bump, it produces a shorter vector
that renormalises straight back - so fading a band out is impossible in that
representation. Normals are now combined in *slope* space: converted, weighted,
summed, converted back. Slopes scale. It is also closer to correct than the
sum-and-renormalise ADR-020 settled for, since heights adding in world space
means world-space slopes add - the same argument as ADR-021's normal
composition.

### 4. Underwater

Two things carry it, and both are physics rather than a blue filter.

**Beer-Lambert absorption, per channel.** Red is absorbed about ten times
faster than blue. A blue fog colour gets the hue but not the behaviour: real
absorption changes the *ratio* between channels with distance, so contrast
collapses toward monochrome rather than toward a tint.

**Snell's window.** From below, the entire 180-degree sky refracts into a cone
about 97 degrees wide overhead; outside it the surface is a perfect mirror by
total internal reflection. Across a wavy surface that boundary breaks into a
bright/dark mosaic - every dark patch is a piece of surface steep enough to
have passed the critical angle. It is a different shading model rather than the
same one with the normal flipped, and the branch falls out of the physics:
`refract()` from water into air returns zero exactly at total internal
reflection, which *is* the window boundary, so there is no tuned threshold.
Fresnel for the water-to-air direction rises to 1 at the critical angle, which
is the physical reason the window has a bright rim.

Depth is queried once per frame through `WaterSurface`, so swimming under a
wake counts, not just the swell.

### What was deliberately not built, and why

**Caustics** need a surface to land on and this demo has no sea floor. Adding a
floor is easy; projecting correct caustics onto it is real work, and a faked
one would undercut the point of everything above it.

**A shoreline is architectural, not effortful.** The FFT ocean assumes
*horizontal homogeneity*: one spectrum, one depth, periodic over the patch.
Shoaling is waves responding to a depth that varies with position, which that
assumption forbids - it cannot be obtained from an FFT patch by tuning
parameters. ADR-019 added finite depth, but *per patch*, which models a
uniformly shallow sea rather than a beach. Real shoaling needs either a
spatially varying spectrum blended between depth zones, or a separate
shallow-water/Boussinesq solver near shore coupled to the FFT offshore.

Breaking is further still. The linear model can *fold* - the Jacobian goes
negative, which is exactly what drives foam - but it cannot *overturn*, because
overturning is where linearisation stops being true. Anything that looks like a
breaking wave here is a fold being shaded as one.

### A build bug that made shader edits do nothing

The SPIR-V rule listed only each shader's own source in `DEPENDS`, not the
`common.glsl` every shader `#include`s and `glslc` resolves itself. Editing the
shared include produced "ninja: no work to do" and the old SPIR-V kept running,
so a shader change appeared to have no effect - and the first round of sky
tuning was silently discarded because of it. Every `.glsl` in the directory is
now a dependency of every shader.

Related trap, from adding uniforms for the detail fade: a `std140` block is
matched by **offset, not by name**. A field inserted at a different position in
the C++ struct than in the GLSL block silently shifts everything after it and
the shader reads the wrong `vec4`, with no validation error, because every
binding is still legitimately bound.

---

## ADR-023 - Unresolved waves become roughness, not nothing

Viewer-only. ADR-022 added a distance fade that drops a cascade once its texels
project to less than a pixel, because what reaches the screen at that point is
not detail but aliasing. That is the right call for *geometry* and it stopped
the horizon shimmering. It also threw something away that it should not have.

The waves in a faded band do not stop existing. Their slopes are what break a
distant sun path into a broad sheen instead of a hard mirror, and deleting the
band deletes that too - so the fade traded a shimmer for a horizon smoother
than open water has any right to be. Measured on the sun road at 12 m/s, the
glitter thinned into isolated specks and died well short of the horizon.

A microfacet BRDF is exactly the tool for surface detail below the sampling
rate: it carries that detail statistically, as a slope distribution, rather
than geometrically. So the variance the fade removes is not discarded, it is
**moved from the mesh into the lobe**. This is the same fact about the sea,
expressed at the scale the renderer can actually carry it.

### The arithmetic, which is exact rather than fitted

Variances of independent contributions add. The fade scales slope *amplitude*,
so it scales variance by its square. And for the Beckmann slope distribution
that GGX is matched to through `alpha`, the total slope variance over both axes
**is** `alpha^2`. Hence

    alpha^2_effective = alpha^2_base + sum_i (1 - fade_i^2) * sigma^2_i

which is the standard LEAN / Toksvig result, and is exact for Gaussian slopes
rather than a curve fitted to look right.

**No Toksvig `|N|` term is needed, and that is a consequence of ADR-022's
design rather than an oversight.** Toksvig's factor recovers variance lost to
mip-filtering a normal map. These cascade textures have no mips at all
(`mipLevels = 1`, `maxLod = 0`): band fading is this renderer's entire LOD
mechanism, so the fade weights account for all of the lost variance and none of
it is hiding in a shortened average normal.

### sigma^2 is measured, and measured only once

Measured from the library's **normal buffer**, not integrated from the spectrum
in closed form. The published normals are the exact displaced-surface normals
with choppy displacement included (ADR-005); a closed-form spectral integral
would describe the *undisplaced* Gaussian surface and disagree with the surface
actually on screen. Agreeing with what is drawn is the entire point.

Measured **once per sea state**, and that is a property of the model rather
than an optimisation: for a stationary Gaussian sea the slope variance is a
spectral integral, the sum of `k^2 S(k)`, which carries no time dependence. The
sea surface moves; its slope statistics do not. It is re-measured only when the
stack is rebuilt, which is on a choppiness change - choppy displacement does
alter the displaced surface's slopes.

At 12 m/s the three cascades measure mean-square slope 0.00663 (800 m), 0.01218
(150 m) and 0.01966 (25 m), totalling 0.0385 - so a fully unresolved sea
reaches `alpha = 0.196` against the base `alpha = 0.0030`, a 65x wider lobe.

**Against measurement, this is an underestimate, and knowably so.** Cox & Munk's
empirical fit gives `sigma^2 = 0.003 + 5.12e-3 * U`, or 0.064 at 12 m/s; this
sea reports 60% of that. The missing 40% is real and accounted for: the finest
cascade cuts off at 0.5 m while Cox & Munk measured a sea carrying capillary
waves down to millimetres. The number is right for the sea being simulated,
which is the one the BRDF has to describe.

### What it changes, and where it deliberately changes nothing

Per-band mean absolute difference over the frame, 12 m/s, against `--no-slope-var`:

| band                  | mean \|delta\| | max | bright pixels off -> on |
|-----------------------|---------------|-----|-------------------------|
| foreground            | **0.00**      | 1   | 0.38% -> 0.38%          |
| near                  | 0.23          | 65  | 0.23% -> 0.24%          |
| mid                   | 6.29          | 95  | 0.76% -> **1.32%**      |
| distant               | 2.78          | 57  | 0.36% -> 0.44%          |
| far (at the horizon)  | 1.64          | 34  | 0.39% -> 0.52%          |

The foreground is **bit-for-bit unchanged**, which is the correctness property
worth stating plainly: where every cascade is fully resolved the sum is zero,
the lobe is the near-mirror it always was, and the term costs nothing it has
not earned. The effect is concentrated precisely in the band where the fade is
partial - which is where the aliasing was.

`--no-slope-var` is the A/B control, in the same spirit as `--no-detail-fade`.
The two compose correctly and independently: with the fade off, every weight is
1, no variance is lost, and no roughness is added.

**Stated limitation.** Only the sun's specular lobe is roughened. The sky
reflection still samples a single mirror direction, so a rough distant sea
reflects the sky more sharply than it should. Doing it properly needs a
prefiltered environment, and the analytic Preetham sky (ADR-022) is evaluated
per-pixel rather than stored in a map there is anything to prefilter. The
Fresnel term is likewise evaluated at the mean normal rather than averaged over
the slope distribution, which is a second-order error at these roughnesses.

---

## ADR-024 - The boat pushes back, and two bugs found by making it

Viewer-only, and all three parts came out of one question: does the demo's boat
actually obey the library's own model of the water, or does it just sit on it?

### 1. The hull as a source, and why it is ONE source

The library documents `SourceKind::Continuous` as "an ongoing condition: a hull
pushing water", and `Obstruction::Neumann` as "correct for hulls". Nothing used
either. The README's claim that objects disturb the water was carried entirely
by thrown rocks, while the flagship floating object was a pure reader.

The hull now submits a disturbance driven by **closing speed**: how fast it is
moving down relative to the water beneath it. That is the same quantity that
lets drag be measured against the water rather than the world, and both
velocities are already in hand, so it costs no extra sampling.

**Which source kind applies is decided by measurement.** A sustained push is
`Continuous`, and it produces almost nothing - correctly. Measured against this
viewer's own 64 m / 0.25 m field, a 0.05 m/s drive (the hull's actual mean)
settles at a peak of **0.24 mm**: a slow push does not pile water up, it makes
it flow, so the wave equation radiates the drive away about as fast as it
arrives. Delivered as an impulse instead, the same total displacement is ~300x
more effective. That asymmetry is why a moored boat bobbing on a swell leaves
the water around it visibly flat.

The term is kept because it is the physically right one and costs one queue
push. It is **not** scaled up to make it show; the ~60x gain that would take is
not a hull pushing water any more. What *is* visible is the part that is
genuinely impulsive - a section that was clear of the water re-entering it fast
- and that gets an `Impulse`, the same treatment as the rock.

**Obstruction is deliberately not used.** `set_obstruction` rebuilds a
nearest-fluid table and the header says it is not a per-frame call; the
viewer's field chases what the camera is looking at, so a world-space mask
would need rebuilding on almost every frame. The right tool for a mask is a
static pier, not a demo whose grid follows the view.

**One centred source, not one per probe - and this is the load-bearing part.**
Four corner sources put the emitted ripple (~2.6 m, at a radius of half the
beam) at almost exactly the hull's 2.15 m beam, so the ripple the boat had just
made arrived at its two beam probes with *opposite sign*. That is a roll torque
the boat applies to itself, and it is regenerative: over four minutes it drove
roll into its +-40 degree clamp in half of the 30 s windows measured, and
widened the hull's travel from 28-108% submerged to -5-136%. Four point probes
alias their own wake.

A single centred source cannot do that. Its ripple is radially symmetric about
the hull centre, so it reaches all four probes alike - a small symmetric heave
term, which is a real added-mass-like effect and self-limiting, and essentially
no pitch or roll torque. Re-measured over 240 s: peak eta settles at 0.05-0.10 m
with no trend across eight windows, roll 9-17 degrees, pitch 8-15 degrees, no
clamping. What is given up is telling a bow slam from a stern one, which at a
0.5-4 m accurate band against a 9 m hull could not be represented faithfully
anyway.

The boat still *reads* the whole field, so a rock dropped alongside rocks it
exactly as before.

### 2. The boat had four centimetres of freeboard

The hull's sizing came from the bounding box of its shell meshes, and the
waterline was set to a fraction of that box's half-height. For a double-ender
that is the wrong ruler, for the same reason `kHullShellPrefixes` exists at
all: the gunwale sweeps up toward both ends, so the box measures the hull where
it is **deepest**, not where it gets wet.

Measured: box depth 1.39 m against an amidships depth of 0.978 m - the stem
posts inflate it by 42%. The old formula put the waterline at +0.243 m from the
hull centre while the amidships gunwale sits at +0.283 m. **Four centimetres of
freeboard**, which is why any real sea closed over the boat rather than being
ridden by it, and why screenshots kept catching it swamped.

The waterline is now taken from the middle fifth of the hull's length: 55% of
that section's depth submerged, giving a draft of 0.538 m and a freeboard of
0.440 m.

### 3. The heave spring was twice as stiff as the hull

Submersion was measured from the hull's vertical **centre**, which is not a
waterline. The resulting equilibrium depth was 0.243 m on a hull that actually
draws 0.538 m, so the restoring spring came out 2.2x too stiff and a 9 m boat
heaved with a 0.99 s period.

Measuring from the keel makes the stiffness `g/draft`, which is the correct
value for a wall-sided hull floating at that draft, and it now falls out of the
geometry rather than being chosen to land the boat at a height. The period is
1.47 s.

### 4. Two shading bugs, one of them mine to have avoided

The prop shader flipped normals on `dot(N, V) < 0`. That is not the same test
as back-facing and it is wrong on curved geometry: an *interpolated* normal on
a smooth hull tips past 90 degrees from the eye well before the true
silhouette, so genuine front faces near the edge were flipped and shaded
inside-out. That was the dark rim around the hull. `gl_FrontFacing` is the
winding the rasteriser actually saw, and is the correct test.

The sail is a doubled sheet whose normals sum to exactly zero, and it rendered
flat sky-grey from **both** sides - 57% of it computed `N.L == 0` even with the
sun directly behind the camera. Thin canvas passes light, so it now gets a
Lambertian transmission term on the reversed normal: the cheapest honest stand-
in for transmission through a thin sheet, and the one lighting condition a sail
is most recognisable in.

The hull also carries a wet band, and it runs **upward** from the waterline
rather than down. Everything below the line is hidden behind the ocean surface
itself, so wetting what is submerged changes nothing anyone can see; the
visible cue is the strip just above the line that the sea keeps washing over.
The band is fitted from the same four probes that float the hull, so it tilts
with the wave rather than cutting the hull dead level - on a 9 m hull in a real
swell that difference is a good fraction of the freeboard.

---

## ADR-025 - Mesh LOD: a geometry clipmap, decoupled from cascade scale

Viewer-only. The mesh was a flat grid tiled over cascade 0's patch: 256 quads
per tile, 7x7 tiles, **6.42M triangles every frame** at a uniform 3.1 m spacing
from under the camera to the far edge. That is the wrong distribution at both
ends — too coarse to resolve a wave three metres away, and far too fine for
water a kilometre out where a whole quad lands inside one pixel.

ADR-022's distance fade addressed the *shading* half of that (aliasing) and
ADR-023 recovered the slope variance it dropped. Neither touched the geometry,
which is where the cost actually was: measured per pass, the ocean draw was
**62% of the frame and about 80% vertex-bound**.

### LOD and cascade scale are different axes

The old mesh conflated them. A cascade is a band of **wavelengths**; a ring is
a band of **distances**. Tiling the largest cascade's patch made the visible
range a function of the spectrum's longest wave, which has nothing to do with
how far the camera can see — and made mesh density uniform, which has nothing
to do with anything.

Rings are placed from the camera and sample every cascade at whatever world
position they land on. The two axes never interact again.

### The structure

Ring 0 is a solid 64x64 grid at `base_cell` metres. Ring i is an annulus with
a centred hole exactly sized for ring i-1 to nest into, at twice the cell size
and twice the footprint. Eight rings at a 0.5 m finest cell reach 2048 m.

Each ring snaps to **its own** cell size, which is the finest snap available at
that scale and what stops near water swimming as the camera moves. The price is
that neighbouring rings are then generally not aligned with each other — ring
i's hole edge and ring i-1's actual outer edge can differ by up to half of ring
i's cell. That offset is a continuous value, not one of a few discrete cases,
so it is closed by a **stitch band** whose topology is fixed and whose
positions are regenerated each frame into a preallocated buffer.

### Cracks are removed, not hidden

At a ring boundary the finer side has twice the vertices, so its surface
follows the water between the coarse vertices while the coarse side draws a
straight segment — a T-junction, and on displaced water it opens into a visible
crack rather than a subtle seam.

The fix is geomorphing, not a skirt. Across the outer fraction of each ring,
vertices blend toward where the *coarser* ring would sample: its cells are
twice as wide, so its nearest grid line is the local position rounded to the
nearest even integer. At full blend the odd vertices land exactly on their even
neighbours, the triangles between them go degenerate, and the boundary becomes
the same straight segment the coarser ring draws. The two sides agree by
construction instead of by a skirt hiding the disagreement.

Chebyshev distance drives the blend, not Euclidean: a ring is a square, and
`length()` would inscribe a circular morph band in a square boundary, leaving
the corners to reach the edge un-morphed and pop.

### Measured

Minimum over 500 frames, RTX 4070 Laptop at 1600x900, MSAA 4x and bloom on:

| | tiled grid | clipmap |
|---|---|---|
| triangles | 6.42M | **54k** (119x fewer) |
| ocean pass | 1.865 ms | **0.279 ms** (6.7x) |
| whole frame | 2.684 ms | **1.085 ms** (2.5x) |
| cell under the camera | 3.1 m | **0.5 m** |
| reach | 2800 m | 2048 m |

It is worth being clear that this is not a quality-for-speed trade. The near
water is **six times finer** than the mesh it replaced while costing a
hundredth of the triangles; the only thing given up is raw reach, and the
horizon fades into the sky long before the mesh ends.

### Six rings was not enough, and only altitude showed it

The default was first set to six rings — 512 m — which looks perfect from the
deck. From 120 m up it is plainly broken: the mesh's far edge stands as a hard
curved boundary against the sky, because the distance fog has not closed by the
time the water runs out. Every ring doubles the reach for about 6k triangles,
so the default is eight and the fix cost 13k triangles out of a 119x saving.

The general lesson is the one this project keeps relearning: a renderer checked
from one camera position has been checked at one camera position. The failure
was invisible at eye level and unmissable from above.

### What this does not do

The reach is still finite. At high enough altitude eight rings will show the
same edge six did, and the honest fix at that point is a coarse horizon skirt
rather than more doublings. Nothing here is adaptive to the view frustum
either: rings are concentric squares centred on the camera, so a substantial
fraction of every ring is behind it. Frustum-culling ring quadrants would cut
the remaining vertex cost further, and at 0.279 ms it is not yet worth the
complexity.

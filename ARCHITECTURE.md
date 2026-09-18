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

## ADR-021 — Mesh LOD: a geometry clipmap, decoupled from cascade scale

### The problem

ADR-015 already named this limitation: every tile in the viewer's mesh draws
at the same resolution regardless of distance, so distant triangles collapse
to sub-pixel size and waste vertex work for nothing. At defaults (256 quads
per tile, 7x7 tiles) that is 6.42M triangles, the overwhelming majority of
them past the point of contributing a visible pixel.

### Decision: concentric clipmap rings, geometry decoupled from cascade scale

A geometry clipmap: `R` concentric square rings centred on the camera, each
ring twice the previous ring's cell spacing and twice its world footprint,
each ring holding a *constant* vertex count regardless of level. That bounds
total triangle count to `O(R * m^2)` rather than scaling with visible area,
which is the entire value of the technique — the alternatives compared
against it (screen-space projected grid, quadtree + geomorphing) are recorded
in the discussion that led here; the projected grid was the strongest
runner-up (least code, cheapest at runtime) but ties triangle allocation to
what is on screen for a free-fly camera that can pitch toward straight-down,
which a fixed ring budget does not.

**Ring geometry is sized independently of the three cascade patch lengths
(800/150/25 m), not aligned to them.** Cascade patch lengths were chosen in
ADR-020 specifically *without* common factors, so that the cascades'
periodic tiling never re-aligns into a visible repeat. A clipmap ring
progression wants the opposite property — a clean power-of-two doubling, so
that ring boundaries nest exactly and adjacent rings differ by a single,
predictable factor. Forcing one scheme to serve both goals would fight both
of them for no benefit. The two concerns are genuinely independent: ring
geometry answers "how many triangles, and where", cascade sampling answers
"which wavelength bands still matter here" — both read the same
distance-from-camera value, but neither needs to share the other's units.

### Why this is simpler than a classic geometry clipmap

The textbook technique (Losasso & Hoppe) is complicated primarily because it
streams a heightmap: each ring mip-maps a different, actually-smoothed copy
of terrain height data, and keeping that data resident under a moving camera
needs toroidal updates - only the thin border that just entered a ring's
footprint is re-fetched each frame, because re-uploading a whole ring's
texture data every frame would be far too much bandwidth.

None of that applies here. There is no heightmap. Every vertex's position is
computed live in the vertex shader by sampling the cascade textures at that
vertex's world (x, z) - exactly what the single-tile mesh already does today.
A ring's vertex and index buffers are therefore fully static, built once at
init exactly like today's mesh, and the only thing that changes per frame is
the ring's world-space centre offset, uploaded the same way the existing
per-tile offset already is. There is no texture data to stream, so there is
no toroidal update to implement at all.

One consequence worth stating plainly: because every ring samples the exact
same cascade textures at the exact same world position, **a vertex's height
never disagrees between rings.** The signal is identical everywhere; only the
mesh's sampling *density* of that signal changes with ring level. This rules
out the classic clipmap "height pop" (where a coarser mip genuinely holds a
different, filtered height) by construction. What is left to solve is purely
geometric: the T-junction where a fine ring's densely-spaced boundary meets a
coarse ring's sparsely-spaced one, and the coarser *effective sample rate*
itself, which can make a fine ripple vanish or appear abruptly as a ring
boundary sweeps across it even with the crack closed.

### Boundary handling: trim strips + a geomorphing blend band

Three options were weighed against the "no visible popping" requirement:
skirts alone (hide the crack, cheapest, does not address the density-change
pop at all), trim strips alone (close the crack exactly, but the sampling
density still jumps discontinuously at the boundary, so fine detail can still
appear/disappear abruptly), and trim strips plus a geomorphing blend band
(closes the crack and fades the transition). Only the third actually
satisfies "no visible popping" rather than "no visible crack" - the other two
were rejected on that basis, not on cost.

**Trim-strip orientation varies per frame, per ring.** Each ring's centre
snaps independently to a multiple of *its own* cell size, which is what
gives every ring the finest possible camera-following precision available
at its own scale. That independence means the offset between a ring and its
coarser neighbour is not fixed: it can land on either side of centre, in x
and independently in z, so the crack the trim strip closes can be on either
of two edges in each axis. One trim shape, built once, is reused for all of
that: it is drawn against two of the hole's four edges - selected per ring
per frame from the sign of that ring's offset relative to its coarser
neighbour's - and mirrored in x and/or z via a sign flip on the same
local-to-world multiply the vertex shader already does, rather than needing
four separately-built rotated meshes. (An earlier draft of this ADR proposed
snapping every ring to its *coarser* neighbour's cell size instead, to fix
the parity and avoid mirroring altogether. That does not actually work: it
only bounds the offset between adjacent centres to a multiple of the coarser
cell size, not to zero, so the ambiguity - and the need to select an
orientation - remains. Corrected before implementation.)

### Implementation and a real bug caught by looking at it

Built as: `clipmap.hpp/.cpp` (host-side ring layout, mesh and stitch-band
generation, no Vulkan types - independently readable, per this file's
established pattern of keeping geometry math out of graphics-API noise),
`RingPush` (a 16-byte push constant: ring offset, cell size, morph start),
and `ocean_view.cpp` drawing one solid centre plus, per outer ring, an
annulus draw and a stitch-band draw against the next-finer ring.

At default settings (6 rings, 0.5 m finest cell), the clipmap draws **~40k
triangles/frame** against the old single-tile mesh's 6.42M at the same
visible range - matching the whole reason for the technique (ADR-015's known
limitation is now fixed).

Verified the same way ADR-015 verifies the rest of this viewer: by looking
at it, not by assuming it compiled. A wireframe screenshot immediately
showed a real defect - a thin bright crack running across the surface,
clearly visible against the sky's fog colour. It survived disabling the
stitch band entirely and separately forcing the geomorph blend to zero,
which ruled out both as the cause and pointed at the one piece neither
bisection touched: the fine-cascade distance fade (`vFade1`/`vFade2`) was
driven by `ring.cellSize`, a per-ring push-constant **that doubles at every
ring boundary**. Since a faded cascade's displacement is scaled by that
factor before being summed into height, two adjacent rings - which must
agree on height at their shared boundary for the surface to be seamless -
were computing genuinely different heights there, because they disagreed
on how much of cascades 1 and 2 to include. No amount of correct stitching
or morphing closes a gap caused by the two sides legitimately disagreeing
about where the surface *is*.

Fixed by driving the fade from world-space distance from the camera
instead: distance is continuous across the whole clipmap, including exactly
at ring seams, so it has no jump for the fade to inherit. Confirmed fixed by
the same means the bug was found - wireframe and shaded screenshots at ring
counts 1 (no stitching at all), 6 (default) and 8 (max), all clean.

This is the reason the geomorph-blend correction earlier in this ADR and
this fade bug are both recorded rather than silently fixed: a technique this
fiddly to get exactly right is exactly where "it compiled" and "it is
correct" come apart, and the second one only gets checked by looking.

### Measured

BENCHMARKS.md has the full comparison against the old single tiled mesh, on
both a discrete and an integrated GPU. Summary: **157x fewer triangles**
(6.42M to 40.8k) at the same visible range, and **3.08x-3.34x less GPU time**
depending on which GPU and which statistic (median vs best) is read - the
two GPUs turn out to need different statistics read for different, both
honest, reasons, explained there.

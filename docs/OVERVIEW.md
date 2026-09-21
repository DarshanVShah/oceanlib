# oceanlib — overview

## What it is

A dependency-free C++20 library for real-time ocean simulation, plus a Vulkan
reference viewer. The core has **zero external dependencies** — its own FFT, its
own PRNG, its own sine/cosine. doctest is test-only; Vulkan and GLFW are
viewer-only.

It runs the whole simulation on the **CPU**, deliberately.

## What it accomplishes

Four promises, each load-bearing rather than aspirational:

1. **Runs on anything.** Multithreaded SIMD CPU FFT — scalar, SSE2, AVX2 and
   NEON, selected at runtime from one binary. No compute shaders required, ever.
2. **Physics matches visuals exactly.** `height_at(x, z)` returns what the
   renderer actually draws at that point, with no GPU readback. Choppy
   displacement moves vertices sideways, so this is a fixed-point inversion, not
   a lookup — a naive lookup is wrong by **725 mm against 535 mm RMS waves**,
   worst at the crests where a boat would notice.
3. **Deterministic.** Same seed and time give identical waves on every machine,
   bit for bit.
4. **Integrates in two memcpys.** Output buffers are already laid out as RGBA32F
   textures.

**Feature surface.** JONSWAP directional spectrum with optional finite-depth
dispersion; Tessendorf choppy displacement with exact displaced-surface normals
and Jacobian foam; multi-scale cascades; 3-D orbital velocity; persistent foam
that advects with the surface current; local interaction (drop a rock, ripples
propagate, disperse, reflect off hulls); world-space height/normal/velocity/foam
queries; a C API with struct versioning for Rust, C# and engine bindings.

**Measured.** 512² ocean in ~1.3 ms threaded. Interaction field 256² in 0.19 ms.
A combined query costs 0.115 µs — only **3.5% more** than the FFT query alone.
174 test cases, 1.14 M assertions.

## Who it helps

- **Game and engine developers** who need buoyancy and gameplay physics that
  agree with what is on screen — boats, swimmers, floating debris — without
  reading back from the GPU.
- **Teams on constrained hardware**: Switch-class consoles, older mobile,
  CPU-only servers, anything without reliable compute shaders.
- **Anyone needing determinism**: lockstep multiplayer, replays,
  server-authoritative physics, visual regression testing.
- **Non-C++ hosts**, through a C ABI designed so an old binary keeps working
  against a new library.

## What is genuinely unusual about it

**Bit-exactness as a hard constraint, including refusing to be more accurate.**
Every SIMD kernel must match scalar to the last bit. FMA is *banned* — not
because it is slower but because it rounds once instead of twice, and a vector
path more accurate than the scalar one would make the ocean depend on which CPU
rendered it. Verified by `memcmp` across every kernel in the build; NEON was
validated under QEMU.

**Physics claims are measured, and the measurements are published.** The ripple
solver's group velocity is checked against deep-water theory: **0.62–0.97%
error** in band. Vertical orbital velocity matches ∂h/∂t to **2.5×10⁻⁵**.
Deep-water orbits come out circular to **1.0000**. Foam decay matches its
analytic exponential to **0.000%**, and its advection is bit-identical to a
hand-shifted field.

**It found a real defect in the published method it is based on.** Tessendorf's
iWave kernel has a **negative Fourier symbol in the aliased corners of k-space
at every radius from 2 to 12** — ω² < 0, an exponentially growing mode that
rounding noise seeds and damping cannot remove. It would surface much later as
"the water sometimes explodes." This ships a band-fitted kernel that is
non-negative everywhere *and* halves the dispersion error.

**Benchmarks catch bugs, not just report speeds.** Twice now. A 256² grid
measured 4× slower *per cell* than 512² — impossible for work proportional to
cell count. The cause was denormal stalls in the absorbing layer; fixing it gave
**6.8×**.

**The documentation states what is approximate.** Twenty-two architecture
decision records separate what is exact (height superposition, displaced-surface
normals) from what is a stated approximation (cascade normal blending,
parameter-space foam advection), and record the limits — no shoaling, no
breaking, no caustics — with the architectural reason each one is not a matter
of effort.

---

Full reasoning is in [ARCHITECTURE.md](../ARCHITECTURE.md); all measured numbers
and their methodology are in [BENCHMARKS.md](../BENCHMARKS.md).

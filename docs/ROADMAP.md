# Roadmap

Two workstreams. The viewer comes first, and not for cosmetic reasons: the
Godot addon does not reuse the Vulkan renderer at all — Godot has its own —
so what crosses over is the *shading model as a specification*. The viewer is
where that model gets designed and proven. Porting it before it is settled
would mean porting it twice.

The target is a renderer that scales: ideal quality on a desktop GPU, and a
measured frame budget at every tier down to hardware with no compute shaders
— the same constituency the library itself was written for (ADR-001).

---

## Workstream A — the viewer

### Phase 0 · GPU timing  — **done**

Per-pass timestamp queries and a frame breakdown. `BENCHMARKS.md` measures the
CPU thoroughly and the GPU not at all, so every statement below about cost is
currently a guess. This makes the rest measurable, and it is also the before /
after evidence.

**If this shows the frame is vertex-bound at 6.42M triangles, Phase 8 moves to
the front.** That is the one result that reorders the plan.

#### A first pass at the results, which was wrong

The first numbers taken here were single samples: one frame's timestamps, read
at the end of a capture run. They gave `upload 2.24 ms` against a 5.46 ms
frame — 41% — and a byte sweep that looked convincingly linear at 3.26, 3.29,
3.17 and 3.28 GB/s across four configurations.

All four of those had been sampled in the same throttled clock state. Running
the *identical* configuration three times gave upload = 2.29, 0.61 and 2.51 ms:
a 4x spread, bimodal, on a laptop GPU whose clocks move underneath the
measurement. Four consistent readings of a quantity that varies 4x run to run
is not corroboration, it is four draws from the same mode.

**One frame's timestamp is not a measurement.** The instrumentation now
accumulates mean, min and max across every frame after a 60-frame warm-up, and
the statistic worth comparing turns out to be the **minimum**: across separate
runs the means disagree by 30% or more while the minima repeat to three decimal
places (upload 0.575 / 0.574, ocean 1.548 / 1.546, sky 0.071 / 0.070). That is
what one would expect — contention and downclocking can only ever *add* time,
so the floor is the clean signal and everything above it is noise.

#### Results

RTX 4070 Laptop, 1600x900, 3 cascades at 256, minimum over 500 frames:

| span   | min ms | share |
|--------|--------|-------|
| ocean — vertex   | 1.377 | **62%** |
| upload           | 0.575 | 26% |
| ocean — fragment | 0.166 | 7% |
| sky              | 0.070 | 3% |
| post             | 0.017 | 1% |
| props            | 0.001 | — |
| **total**        | **2.225** | |

Mesh density with everything else fixed splits the ocean pass:

| mesh | triangles | ocean |
|------|-----------|-------|
| 256  | 6.42M     | 1.543 ms |
| 128  | 1.61M     | 0.525 ms |
| 64   | 0.40M     | 0.241 ms |
| 32   | 0.10M     | 0.166 ms |

`upload` reads 0.576-0.586 ms across all four, as it must — it does not care
what is drawn. The 0.10M row is essentially pure fragment work, so the vertex
half of the full-density pass is about 1.38 ms and the fragment half about
0.17 ms.

Uploaded bytes, swept independently:

| uploaded per frame | upload | rate |
|--------------------|--------|------|
| 7.34 MB | 0.575 ms | 12.77 GB/s |
| 3.15 MB | 0.248 ms | 12.70 GB/s |
| 2.62 MB | 0.211 ms | 12.42 GB/s |
| 6.36 MB | 0.503 ms | 12.64 GB/s |

Still exactly linear in bytes, but at **12.6 GB/s** — an ordinary PCIe rate,
not the 3.2 GB/s "bandwidth wall" the first pass reported. The linearity was
real; the rate was an artifact of measuring throttled frames.

**Conclusions.**

1. **The ocean pass is 62% vertex throughput.** Clipmap LOD was scheduled last
   and is the single biggest item in the frame. It moves to the front.
2. **The upload is 26%** — smaller than first claimed, still second-largest,
   and halving its bytes halves its cost exactly.
3. **The entire fragment shader is 7%.** Every per-pixel cost in this renderer
   put together — three Preetham evaluations, six texture fetches, the whole
   BRDF — is a fifteenth of the frame.

The third one is the useful one, and the correction strengthens it rather than
weakening it: **fragment cost is not the problem, so the shading can be far
more ambitious than planned.** Bloom, SSR and a better BRDF are charged
against the cheapest 7% of the frame. The pretty things are affordable; the
triangles and the bus are not.

Caveat: one desktop GPU at 1600x900. Lower-end and mobile parts shift the
balance toward fragment — and toward upload and vertex too — so the ranking is
unlikely to invert, but Phase 9 measures rather than assumes.

#### Revised order

Phases 1-3 stand: Phase 1 gates everything after it, and 2-3 are the visual
payoff the fragment budget can clearly afford. Then, reordered by the numbers
above:

- **Phase 4a · Clipmap LOD** — **done**, see ADR-025. 1.38 ms of vertex work on
  triangles that are mostly sub-pixel, and the largest single item in the frame.
- **Phase 4b · Halve the upload** — **done**, see ADR-026. Predicted 0.29 ms
  and delivered 0.278. The part that was not predicted: pricing the CPU side of
  the trade turned up a pre-existing 0.9 ms/frame defect in the foam alpha
  patch, so the CPU fill got *faster* too rather than paying for the saving.
- **Phase 4c · Prefiltered sky**, demoted from Phase 4. Worth doing for the
  roughness-aware reflection it enables — ADR-023's stated limitation — which
  is a *quality* argument. It was sold on performance, and at 7% fragment cost
  that argument does not survive contact with the measurement.

### Where it stands

Phases 0-4b are done. Frame budget at the default quality — clipmap LOD, MSAA
4x, bloom, ACES, RGBA16F upload — minimum over 500 frames at 1600x900:

| span | min ms | share |
|------|--------|-------|
| **upload** | **0.297** | **41%** |
| ocean  | 0.200 | 28% |
| bloom  | 0.110 | 15% |
| sky    | 0.084 | 12% |
| post   | 0.014 | 2% |
| **total** | **0.721** | |

After ADR-027's shading work the ocean pass is 0.272 ms and the total 0.811 ms.

Plus 0.328 ms of CPU preparing the staging buffer, which is reported alongside
because an optimisation that moves work between processors is not honestly
described by a number from only one of them.

The frame started this work at **2.225 ms with nothing on it**. It now costs
**0.721 ms with MSAA 4x, a five-level bloom chain, a filmic tone curve and six
times finer water under the camera** — 3.1x faster while carrying four
features it did not have. The Phase 0 prediction held throughout: everything
added landed in the part of the frame measured as cheap, and the two things
attacked for cost were the two the measurement named.

| | start | now |
|---|---|---|
| geometry (ocean pass) | 62% of frame | 28% |
| upload | 26% | 41% |
| CPU staging fill | 0.572 ms | 0.328 ms |
| triangles/frame | 6.42M | 54k |

The upload is still the largest span, and it is still bandwidth rather than
GPU work. What is left of it is genuinely needed data: the next honest lever is
not sending what has not changed, and after that moving the FFT onto the GPU,
which is a different project and would cost the library its bit-exactness
across backends. Neither is worth doing before the quality work below.

**Phase 7's shading corrections are done** (ADR-027), pulled forward because
they were what the picture actually needed: subsurface scattering, a complete
microfacet BRDF, sun colour from the sky model, and the removal of a second
sun that had been quietly defeating ADR-023. 0.09 ms, taking the frame to
0.811 ms.

Remaining, in measured order: the prefiltered sky (4c), then SSR and shadows —
the boat still does not appear in the water it is floating in, which is now the
most visible thing wrong with the image.

`--msaa`, `--bloom`, `--tonemap`, `--rings` and `--fp32` each turn their
feature off or down completely, so the tier machinery in Phase 9 already has
something to drive.

---

### Phase 1 · HDR off-screen target and a post chain  — **done**

The viewer renders in a single pass straight to the swapchain, with
`tonemap()` called inline at the end of `ocean.frag`, `object.frag` and
`sky.frag`. Nothing that needs to read the finished image can exist in that
structure: no bloom, no screen-space reflection, no temporal AA, no grade.

Scene → `R16G16B16A16_SFLOAT` → post → swapchain, with tonemapping moved out
of the material shaders into the post pass where it belongs. This phase
changes no pixels on purpose; it is the gate the next five wait behind.

### Phase 2 · Bloom and a filmic tonemap  — **done**

The first visible payoff, and the reason Phase 1 exists. Reinhard is leaving
real quality on the table, and sun glitter on water is the exact case bloom
was invented for.

### Phase 3 · MSAA  — **done**

There is no anti-aliasing of any kind today (`VK_SAMPLE_COUNT_1_BIT`
everywhere), which is why the rigging stairsteps in every screenshot. 4x
resolves the geometric edges; specular aliasing is already handled
statistically by ADR-023, so this does not have to carry that load too.

### Phase 4 · Prefiltered sky

`ocean.frag` evaluates the Preetham model **three times per pixel** — once for
the reflection, once for the ambient tint, once for the distance fade — from a
`common.glsl` carrying 13 transcendental call sites. The sky only changes when
the sun or the turbidity moves, which is close to never.

Bake it to a small lat-long map with a mip chain. Two results from one change:
the per-pixel cost collapses to texture fetches, and **the mip chain is the
prefiltered environment** ADR-023 records as its stated limitation — so the
sky reflection becomes roughness-aware instead of a single mirror ray.

### Phase 5 · Screen-space reflections

The boat does not appear in the water under it. That is the largest remaining
"pasted on" cue in the scene, larger than anything the hull's own shading was
getting wrong. Water is the favourable case for SSR: the surface is close to
planar and the reflected ray leaves it immediately.

### Phase 6 · Shadows

The boat casts nothing onto the water. One shadow map over the props.

### Phase 7 · Shading model corrections

- Sun radiance taken from the sky model rather than the hardcoded
  `vec3(1.0, 0.95, 0.86)`. This is the same defect ADR-024 fixed in
  `object.frag` and it is still live in `ocean.frag`: a light with no
  relationship to the sky the water is reflecting.
- Energy conservation. The specular is currently `D` alone, with no geometry
  term and an arbitrary `0.35`.
- Subsurface scattering from depth rather than the `lift * through` stand-in.
- Aerial perspective with sun-direction-dependent in-scatter, instead of one
  fog colour.

### Phase 8 · Clipmap LOD  — **done as Phase 4a (ADR-025)**

6.42M triangles per frame of uniform grid, most of them sub-pixel at distance.
This is the wall for older hardware, and it buys back the budget Phases 2-6
spend. The `lod-clipmap` branch is the starting point.

### Phase 9 · Quality tiers

Ultra through Low, switching SSR, shadows, AA, cascade count, mesh density and
sky resolution. Publish the measured budget per tier — a claim about older
devices is worth what its measurements are worth, and right now "Switch-class
consoles, older mobile" in README.md is measured on neither.

**Deferred: temporal AA.** It would take the residual shimmer MSAA leaves, but
it needs motion vectors and a history buffer, and it interacts badly with a
surface that is displaced every frame. Revisit after Phase 6, and only if the
Phase 0 numbers justify it.

---

## Workstream B — Godot

GDExtension is C++, so it binds `ocean.hpp` / `cascade.hpp` / `foam.hpp`
directly and never touches the C API. That matters: `CascadeStack` and
`FoamField` have **no C API at all**, so a C# or Rust binding would hit that
wall immediately and Godot does not.

- **B1** — `godot-cpp` skeleton. An `OceanSim` node owning `CascadeStack`,
  `FoamField` and `InteractionField`, with the cascades uploaded as a
  `Texture2DArray`.
- **B2** — `height_at` / `sample_at` exposed to GDScript. **This is the reason
  anyone would install it.** Godot has no credible water physics, and buoyancy
  that provably matches the drawn surface is something nothing else offers.
- **B3** — the Phase 1-7 shading model, rewritten as a Godot spatial shader.
- **B4** — a demo project and an Asset Library submission.

The addon ships the shading, not only the simulation. Handing someone a
displacement buffer and wishing them luck is not a thing anyone installs.

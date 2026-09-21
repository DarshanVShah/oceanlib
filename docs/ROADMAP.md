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

### Phase 0 · GPU timing

Per-pass timestamp queries and a frame breakdown. `BENCHMARKS.md` measures the
CPU thoroughly and the GPU not at all, so every statement below about cost is
currently a guess. This makes the rest measurable, and it is also the before /
after evidence.

**If this shows the frame is vertex-bound at 6.42M triangles, Phase 8 moves to
the front.** That is the one result that reorders the plan.

#### Results, and they did reorder it

RTX 4070 Laptop, 1600x900, 3 cascades at 256, default settings:

    gpu: 5.46 ms total, upload 2.24, sky 0.17, ocean 3.05, props 0.01

Sweeping mesh density with everything else fixed separates the ocean pass into
its vertex and fragment halves:

| mesh | triangles | ocean |
|------|-----------|-------|
| 256  | 6.42M     | 3.02 ms |
| 128  | 1.61M     | 1.32 ms |
| 64   | 0.40M     | 0.85 ms |
| 32   | 0.10M     | 0.60 ms |

A 64x reduction in triangles takes the pass from 3.02 ms to 0.60 ms, so about
**2.4 ms is vertex throughput and about 0.6 ms is everything the fragment
shader does** — and `upload` does not move at all, as it should not.

Sweeping the uploaded bytes instead:

| uploaded per frame | upload | implied rate |
|--------------------|--------|--------------|
| 7.34 MB (baseline) | 2.26 ms | 3.26 GB/s |
| 3.15 MB (1 cascade) | 0.96 ms | 3.29 GB/s |
| 2.62 MB (cascades at 128) | 0.83 ms | 3.17 GB/s |
| 6.36 MB (interaction at 64) | 1.94 ms | 3.28 GB/s |

Four configurations, one rate. The upload is a **pure bandwidth wall, exactly
linear in bytes**, with no fixed overhead to amortise away. Halving the bytes
halves the time.

**Three conclusions, none of which the plan had right.**

1. **The upload is 41% of the frame and was not on the roadmap at all.**
   Three cascades of displacement and normals plus the interaction field is
   7.34 MB every frame, 440 MB/s at 60 Hz, and on unified-memory mobile
   hardware that is cache pressure rather than a spare bus.
2. **The ocean pass is ~80% vertex-bound.** Phase 8 was scheduled last and is
   attacking 44% of the frame.
3. **The entire fragment shader is 0.6 ms — 11%.** The three Preetham
   evaluations per pixel that Phase 4 exists to remove are a fraction of that.

The last one is the useful surprise: **fragment cost is not the problem, so
there is real headroom to make the shading much more ambitious.** Bloom, SSR
and a better BRDF are all fragment work being charged against the cheapest
11% of the frame. The quality ambitions are affordable; the triangles and the
bus are not.

Caveat: this is one desktop GPU at 1600x900. Lower-end and mobile parts shift
the balance toward fragment — but they shift it toward upload and vertex too,
so the ranking is unlikely to inverse. Phase 9 is where that gets measured
rather than assumed.

#### Revised order

Phases 1-3 stand: they are the visual payoff the fragment budget can now
clearly afford, and Phase 1 gates everything after it. Then:

- **Phase 4a (new) · Halve the upload.** `R16G16B16A16_SFLOAT` instead of
  `R32G32B32A32_SFLOAT` for the cascade and interaction textures. Directly
  worth ~1.1 ms on the measured rate, the single largest win available, and it
  costs a float-to-half conversion on the CPU that wants measuring against
  what it saves. Also consider skipping the interaction upload entirely while
  the field is quiescent, which is most of the time.
- **Phase 4b · Clipmap LOD**, promoted from Phase 8. 2.4 ms of vertex work on
  triangles that are mostly sub-pixel.
- **Phase 4c · Prefiltered sky**, demoted from Phase 4. Still worth doing for
  the roughness-aware reflection it enables — which is a *quality* argument,
  ADR-023's stated limitation, not the performance argument it was sold on.

### Phase 1 · HDR off-screen target and a post chain

The viewer renders in a single pass straight to the swapchain, with
`tonemap()` called inline at the end of `ocean.frag`, `object.frag` and
`sky.frag`. Nothing that needs to read the finished image can exist in that
structure: no bloom, no screen-space reflection, no temporal AA, no grade.

Scene → `R16G16B16A16_SFLOAT` → post → swapchain, with tonemapping moved out
of the material shaders into the post pass where it belongs. This phase
changes no pixels on purpose; it is the gate the next five wait behind.

### Phase 2 · Bloom and a filmic tonemap

The first visible payoff, and the reason Phase 1 exists. Reinhard is leaving
real quality on the table, and sun glitter on water is the exact case bloom
was invented for.

### Phase 3 · MSAA

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

### Phase 8 · Clipmap LOD

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

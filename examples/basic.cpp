// oceanlib - minimal integration example.
//
// Shows the whole lifecycle an engine needs: describe, construct, update once
// per frame, upload the buffers, and query heights for physics. Builds with
// -DOCEAN_BUILD_EXAMPLES=ON.

#include "ocean/ocean.hpp"

#include <cmath>
#include <cstdio>

int main()
{
    // --- 1. Describe the sea state -----------------------------------------
    ocean::OceanDesc desc;
    desc.size         = 256;    // grid resolution, power of two
    desc.patch_length = 200.0f; // one tile is 200 m across
    desc.seed         = 1337;   // same seed + same time = same ocean, always

    desc.spectrum.wind_speed     = 12.0f;  // m/s at 10 m height
    desc.spectrum.fetch          = 100000.0f;
    desc.spectrum.wind_direction = 0.4f;   // radians from +X

    desc.choppiness = 1.0f;  // Tessendorf's lambda: sharpens crests

    // Leaving desc.parallel_for null selects the built-in thread pool. An
    // engine would instead point it at its own ParallelFor and pass its
    // scheduler through parallel_for_user.

    // --- 2. Construct. Everything allocates here, once. --------------------
    ocean::Ocean sim{desc};

    // --- 3. Per frame ------------------------------------------------------
    for (int frame = 0; frame < 3; ++frame) {
        const double t = frame / 60.0;

        // Absolute time, not a delta: the surface is a pure function of
        // (seed, desc, t), so pausing or seeking reproduces exactly.
        sim.update(t);

        const ocean::Buffers b = sim.buffers();

        // These two pointers are laid out exactly as RGBA32F textures, so a
        // renderer uploads the whole ocean with two calls and no repacking:
        //
        //   glTexSubImage2D(..., GL_RGBA, GL_FLOAT, b.displacement);
        //   glTexSubImage2D(..., GL_RGBA, GL_FLOAT, b.normal);

        // Summarise what the renderer would be drawing.
        const std::size_t cells = static_cast<std::size_t>(b.size) * b.size;
        double sum_sq = 0.0, foam = 0.0;
        float lowest = 1e30f, highest = -1e30f;
        for (std::size_t i = 0; i < cells; ++i) {
            const float h = b.displacement[4 * i + 1];
            sum_sq += static_cast<double>(h) * h;
            lowest  = std::fmin(lowest, h);
            highest = std::fmax(highest, h);
            foam += b.displacement[4 * i + 3];
        }
        // Significant wave height, the standard oceanographic measure: the
        // mean height of the highest third of waves, which works out to
        // 4 * the standard deviation of the surface.
        const double hs = 4.0 * std::sqrt(sum_sq / static_cast<double>(cells));

        std::printf("t=%.3fs  Hs=%.2f m  range=[%.2f, %.2f] m  foam=%.1f%%\n", t,
                    hs, lowest, highest, 100.0 * foam / static_cast<double>(cells));
    }

    // --- 4. Physics queries ------------------------------------------------
    //
    // These agree with what the renderer draws, with no GPU readback. Note
    // this is NOT a texture lookup: choppy displacement moves vertices
    // sideways, so finding the surface above a world position means inverting
    // that displacement.
    std::printf("\nbuoyancy samples at t=%.3fs:\n", sim.time());
    for (int i = 0; i < 5; ++i) {
        const float x = 20.0f * i;
        const float z = 15.0f * i;
        const ocean::Surface s = sim.sample_at(x, z);
        std::printf("  (%6.1f, %6.1f)  height %+6.2f m  normal (%+.2f, %+.2f, %+.2f)  foam %.2f\n",
                    x, z, s.height, s.normal_x, s.normal_y, s.normal_z, s.foam);
    }

    return 0;
}

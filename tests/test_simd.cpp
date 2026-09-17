#include <doctest/doctest.h>

#include "core/cpu_features.hpp"
#include "core/fft.hpp"
#include "core/fft_kernel.hpp"
#include "core/rng.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace ocean::detail;

namespace {

// Every SIMD level this build contains AND this CPU can execute.
std::vector<SimdLevel> available_levels()
{
    std::vector<SimdLevel> levels{SimdLevel::Scalar};
    const SimdLevel cap = max_simd_level();

    if (cap == SimdLevel::Neon) {
        levels.push_back(SimdLevel::Neon);
    } else {
        if (cap == SimdLevel::Sse2 || cap == SimdLevel::Avx2) {
            levels.push_back(SimdLevel::Sse2);
        }
        if (cap == SimdLevel::Avx2) {
            levels.push_back(SimdLevel::Avx2);
        }
    }
    return levels;
}

// doctest stringifies a raw const char* as a pointer, so wrap it.
std::string level_name(SimdLevel l) { return std::string(simd_level_name(l)); }

void fill_random(std::vector<float>& v, std::uint64_t seed)
{
    Pcg32 rng = seed_pcg32(seed, 3);
    fill_gaussians(rng, v.data(), v.size());
}

}  // namespace

TEST_CASE("CPU detection reports a level this machine can actually run")
{
    const SimdLevel cap = max_simd_level();
    MESSAGE("detected SIMD level: " << level_name(cap));
    CHECK(simd_level_name(cap) != nullptr);

    // Whatever we detected, the scalar kernel must always be selectable.
    CHECK(select_stage_kernel(SimdLevel::Scalar) == &stage_scalar);

    // Asking for more than the CPU has must be clamped, never granted.
    const SimdLevel forced = force_simd_level(SimdLevel::Avx2);
    if (cap != SimdLevel::Avx2) {
        CHECK(forced != SimdLevel::Avx2);
    }
    force_simd_level(cap);  // restore
}

TEST_CASE("every SIMD stage kernel is bit-identical to the scalar kernel")
{
    // The load-bearing test for ADR-013.
    //
    // memcmp, not an epsilon comparison, and that is the whole point. If the
    // AVX2 kernel were merely *close* to the scalar one - which is what using
    // an FMA would produce, and an FMA is actually MORE accurate - then the
    // ocean a player sees would depend on which CPU rendered it. Promise #3
    // would silently become "deterministic per machine".
    //
    // Exercised directly at the stage level rather than through a whole
    // transform, so a disagreement points at one stage and one value of s
    // instead of at the end of a nine-stage pipeline.
    for (SimdLevel level : available_levels()) {
        CAPTURE(level_name(level));
        const StageKernel kernel = select_stage_kernel(level);
        if (kernel == &stage_scalar) continue;

        for (std::uint32_t n : {8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u}) {
            CAPTURE(n);
            std::vector<float> xr(n), xi(n), twr(n / 2), twi(n / 2);
            fill_random(xr, 100u + n);
            fill_random(xi, 200u + n);
            fill_random(twr, 300u + n);
            fill_random(twi, 400u + n);

            // Walk every stage the real transform would run, so that both the
            // wide path (s >= vector width) and the scalar tail (s < width)
            // are covered at every size.
            for (std::uint32_t len = n, s = 1; len > 1; len >>= 1, s <<= 1) {
                CAPTURE(len);
                CAPTURE(s);

                std::vector<float> ref_r(n, 0.0f), ref_i(n, 0.0f);
                std::vector<float> simd_r(n, 0.0f), simd_i(n, 0.0f);

                stage_scalar(xr.data(), xi.data(), ref_r.data(), ref_i.data(),
                             twr.data(), twi.data(), len, s, s);
                kernel(xr.data(), xi.data(), simd_r.data(), simd_i.data(),
                       twr.data(), twi.data(), len, s, s);

                REQUIRE(std::memcmp(ref_r.data(), simd_r.data(),
                                    n * sizeof(float)) == 0);
                REQUIRE(std::memcmp(ref_i.data(), simd_i.data(),
                                    n * sizeof(float)) == 0);
            }
        }
    }
}

TEST_CASE("whole transforms agree bit-for-bit across SIMD levels")
{
    for (std::uint32_t n : {16u, 64u, 256u, 1024u}) {
        CAPTURE(n);
        std::vector<float> src_re(n), src_im(n);
        fill_random(src_re, 555u + n);
        fill_random(src_im, 666u + n);

        std::vector<float> ref_re = src_re, ref_im = src_im;
        std::vector<float> scratch(4 * n * kColumnBatch);
        FftPlan scalar_plan{n, SimdLevel::Scalar};
        scalar_plan.transform(ref_re.data(), ref_im.data(), 1, scratch.data(),
                              FftSign::Inverse);

        for (SimdLevel level : available_levels()) {
            CAPTURE(level_name(level));
            std::vector<float> re = src_re, im = src_im;
            FftPlan plan{n, level};
            CHECK(plan.simd_level() == level);
            plan.transform(re.data(), im.data(), 1, scratch.data(),
                           FftSign::Inverse);

            REQUIRE(std::memcmp(ref_re.data(), re.data(), n * sizeof(float)) == 0);
            REQUIRE(std::memcmp(ref_im.data(), im.data(), n * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("2D transforms agree bit-for-bit across SIMD levels")
{
    for (std::uint32_t n : {32u, 128u}) {
        CAPTURE(n);
        const std::size_t cells = static_cast<std::size_t>(n) * n;
        std::vector<float> src_re(cells), src_im(cells);
        fill_random(src_re, 777u + n);
        fill_random(src_im, 888u + n);

        std::vector<float> scratch(4 * n * kColumnBatch);
        std::vector<float> ref_re = src_re, ref_im = src_im;
        FftPlan scalar_plan{n, SimdLevel::Scalar};
        scalar_plan.transform_2d(ref_re.data(), ref_im.data(), scratch.data(),
                                 FftSign::Inverse);

        for (SimdLevel level : available_levels()) {
            CAPTURE(level_name(level));
            std::vector<float> re = src_re, im = src_im;
            FftPlan plan{n, level};
            plan.transform_2d(re.data(), im.data(), scratch.data(),
                              FftSign::Inverse);
            REQUIRE(std::memcmp(ref_re.data(), re.data(), cells * sizeof(float)) == 0);
            REQUIRE(std::memcmp(ref_im.data(), im.data(), cells * sizeof(float)) == 0);
        }
    }
}

TEST_CASE("SIMD kernels handle sizes below the vector width")
{
    // AVX2 processes 8 floats and SSE2 processes 4, so a transform of length 4
    // never enters the wide path at all and runs entirely on the scalar tail.
    // This is the case a kernel written only for the happy path gets wrong.
    for (SimdLevel level : available_levels()) {
        CAPTURE(level_name(level));
        for (std::uint32_t n : {2u, 4u}) {
            CAPTURE(n);
            std::vector<float> a_re(n), a_im(n), scratch(4 * n * kColumnBatch);
            fill_random(a_re, 11u + n);
            fill_random(a_im, 22u + n);
            std::vector<float> b_re = a_re, b_im = a_im;

            FftPlan{n, SimdLevel::Scalar}.transform(a_re.data(), a_im.data(), 1,
                                                    scratch.data(), FftSign::Forward);
            FftPlan{n, level}.transform(b_re.data(), b_im.data(), 1,
                                        scratch.data(), FftSign::Forward);

            REQUIRE(std::memcmp(a_re.data(), b_re.data(), n * sizeof(float)) == 0);
            REQUIRE(std::memcmp(a_im.data(), b_im.data(), n * sizeof(float)) == 0);
        }
    }
}

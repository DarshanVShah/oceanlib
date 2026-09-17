#include <doctest/doctest.h>

#include "core/fft.hpp"
#include "core/rng.hpp"
#include "reference_dft.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace ocean::detail;

namespace {

struct Field {
    std::uint32_t      n = 0;
    std::vector<float> re, im, scratch;

    explicit Field(std::uint32_t size)
        : n(size),
          re(static_cast<std::size_t>(size) * size, 0.0f),
          im(static_cast<std::size_t>(size) * size, 0.0f),
          scratch(4u * static_cast<std::size_t>(size), 0.0f)
    {}
};

// Deterministic pseudo-random complex data, so a failure is always
// reproducible from the test name alone.
void fill_random(std::vector<float>& re, std::vector<float>& im,
                 std::uint64_t seed)
{
    Pcg32 rng = seed_pcg32(seed, 17);
    fill_gaussians(rng, re.data(), re.size());
    fill_gaussians(rng, im.data(), im.size());
}

}  // namespace

// ---------------------------------------------------------------------------
// Convention: pin the sign of the exponent analytically, so we are not merely
// agreeing with our own reference implementation.
// ---------------------------------------------------------------------------

TEST_CASE("forward transform of a pure exponential lands entirely in one bin")
{
    // sum_j exp(2*pi*i*j*k0/N) * exp(-2*pi*i*j*k/N) = N when k == k0, else 0.
    // This pins the sign convention against theory rather than against our own
    // DFT: if both had the sign flipped, the comparison tests below would still
    // pass while the ocean ran backwards.
    constexpr std::uint32_t n  = 64;
    constexpr std::uint32_t k0 = 7;

    std::vector<float> re(n), im(n), scratch(4 * n);
    for (std::uint32_t j = 0; j < n; ++j) {
        const double a = refdft::kTwoPi * k0 * j / n;
        re[j] = static_cast<float>(std::cos(a));
        im[j] = static_cast<float>(std::sin(a));
    }

    FftPlan plan{n};
    plan.transform(re.data(), im.data(), 1, scratch.data(), FftSign::Forward);

    for (std::uint32_t k = 0; k < n; ++k) {
        const float want = (k == k0) ? static_cast<float>(n) : 0.0f;
        CHECK(std::abs(re[k] - want) < 1e-3f);
        CHECK(std::abs(im[k]) < 1e-3f);
    }
}

TEST_CASE("an impulse transforms to a constant, and a constant to an impulse")
{
    constexpr std::uint32_t n = 32;
    FftPlan plan{n};
    std::vector<float> scratch(4 * n);

    SUBCASE("impulse -> constant") {
        std::vector<float> re(n, 0.0f), im(n, 0.0f);
        re[0] = 1.0f;
        plan.transform(re.data(), im.data(), 1, scratch.data(), FftSign::Forward);
        for (std::uint32_t k = 0; k < n; ++k) {
            CHECK(re[k] == doctest::Approx(1.0f).epsilon(1e-5));
            CHECK(std::abs(im[k]) < 1e-5f);
        }
    }
    SUBCASE("constant -> impulse of height N") {
        std::vector<float> re(n, 1.0f), im(n, 0.0f);
        plan.transform(re.data(), im.data(), 1, scratch.data(), FftSign::Forward);
        CHECK(re[0] == doctest::Approx(static_cast<float>(n)).epsilon(1e-5));
        for (std::uint32_t k = 1; k < n; ++k) {
            CHECK(std::abs(re[k]) < 1e-3f);
            CHECK(std::abs(im[k]) < 1e-3f);
        }
    }
}

// ---------------------------------------------------------------------------
// Agreement with the naive DFT.
// ---------------------------------------------------------------------------

TEST_CASE("1D FFT matches the naive DFT")
{
    for (std::uint32_t n : {2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u}) {
        CAPTURE(n);
        std::vector<float> re(n), im(n), scratch(4 * n);
        fill_random(re, im, 0xC0FFEEu + n);

        const std::vector<float> in_re = re;
        const std::vector<float> in_im = im;

        FftPlan plan{n};

        for (int sign : {-1, +1}) {
            CAPTURE(sign);
            re = in_re;
            im = in_im;
            plan.transform(re.data(), im.data(), 1, scratch.data(),
                           sign < 0 ? FftSign::Forward : FftSign::Inverse);

            std::vector<double> ref_re, ref_im;
            refdft::dft_1d(in_re.data(), in_im.data(), n, sign, ref_re, ref_im);

            // Measured on MSVC 19.44 /O2 /fp:precise: 3.6e-8 at N=2 rising to
            // 1.4e-7 at N=256 - i.e. float epsilon (1.19e-7), growing like
            // sqrt(log2 N) as the theory says. The 1e-6 bound leaves ~7x
            // headroom for other libm implementations while still being tight
            // enough that a wrong twiddle index or a flipped sign (both O(1)
            // errors) cannot slip through.
            const double err =
                refdft::relative_error(re.data(), im.data(), ref_re, ref_im);
            CAPTURE(err);
            CHECK(err < 1e-6);
        }
    }
}

TEST_CASE("2D FFT matches the naive O(N^4) DFT")
{
    // Kept small: the reference is a genuine quadruple sum, so even N=32 costs
    // about a million complex multiplies.
    for (std::uint32_t n : {2u, 4u, 8u, 16u, 32u}) {
        CAPTURE(n);
        Field f{n};
        fill_random(f.re, f.im, 0xBEEF00u + n);

        const std::vector<float> in_re = f.re;
        const std::vector<float> in_im = f.im;

        FftPlan plan{n};

        for (int sign : {-1, +1}) {
            CAPTURE(sign);
            f.re = in_re;
            f.im = in_im;
            plan.transform_2d(f.re.data(), f.im.data(), f.scratch.data(),
                              sign < 0 ? FftSign::Forward : FftSign::Inverse);

            std::vector<double> ref_re, ref_im;
            refdft::dft_2d(in_re.data(), in_im.data(), n, sign, ref_re, ref_im);

            // Measured: 4.9e-8 at N=2 rising to 1.4e-7 at N=64.
            const double err = refdft::relative_error(f.re.data(), f.im.data(),
                                                      ref_re, ref_im);
            CAPTURE(err);
            CHECK(err < 1e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// Structural properties.
// ---------------------------------------------------------------------------

TEST_CASE("forward then inverse recovers N*x in 1D")
{
    for (std::uint32_t n : {8u, 64u, 512u}) {
        CAPTURE(n);
        std::vector<float> re(n), im(n), scratch(4 * n);
        fill_random(re, im, 4242u + n);
        const std::vector<float> in_re = re, in_im = im;

        FftPlan plan{n};
        plan.transform(re.data(), im.data(), 1, scratch.data(), FftSign::Forward);
        plan.transform(re.data(), im.data(), 1, scratch.data(), FftSign::Inverse);

        const float inv = 1.0f / static_cast<float>(n);
        for (std::uint32_t j = 0; j < n; ++j) {
            CHECK(std::abs(re[j] * inv - in_re[j]) < 1e-4f);
            CHECK(std::abs(im[j] * inv - in_im[j]) < 1e-4f);
        }
    }
}

TEST_CASE("forward then inverse recovers N^2*x in 2D")
{
    for (std::uint32_t n : {8u, 64u, 128u}) {
        CAPTURE(n);
        Field f{n};
        fill_random(f.re, f.im, 777u + n);
        const std::vector<float> in_re = f.re, in_im = f.im;

        FftPlan plan{n};
        plan.transform_2d(f.re.data(), f.im.data(), f.scratch.data(), FftSign::Forward);
        plan.transform_2d(f.re.data(), f.im.data(), f.scratch.data(), FftSign::Inverse);

        const float inv = 1.0f / static_cast<float>(n * n);
        for (std::size_t i = 0; i < f.re.size(); ++i) {
            CHECK(std::abs(f.re[i] * inv - in_re[i]) < 1e-3f);
            CHECK(std::abs(f.im[i] * inv - in_im[i]) < 1e-3f);
        }
    }
}

TEST_CASE("a Hermitian-symmetric spectrum inverse-transforms to a real field")
{
    // This is the property the entire ocean pipeline rests on. Every spectrum
    // we build satisfies S(-k) = conj(S(k)), and that is exactly what makes the
    // displaced surface real-valued instead of complex nonsense. If this test
    // fails, the imaginary parts we are about to throw away were not zero.
    constexpr std::uint32_t n = 32;
    Field f{n};

    Pcg32 rng = seed_pcg32(31337, 5);
    auto at = [](std::uint32_t x, std::uint32_t y) {
        return static_cast<std::size_t>(y) * n + x;
    };

    for (std::uint32_t y = 0; y < n; ++y) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::uint32_t mx = (n - x) % n;  // index of -k
            const std::uint32_t my = (n - y) % n;
            if (at(x, y) < at(mx, my)) {
                const GaussianPair g = next_gaussian_pair(rng);
                f.re[at(x, y)]   = g.a;
                f.im[at(x, y)]   = g.b;
                f.re[at(mx, my)] = g.a;
                f.im[at(mx, my)] = -g.b;  // conjugate partner
            } else if (at(x, y) == at(mx, my)) {
                // Self-conjugate bins (DC and Nyquist) must be purely real.
                f.re[at(x, y)] = next_gaussian_pair(rng).a;
                f.im[at(x, y)] = 0.0f;
            }
        }
    }

    FftPlan plan{n};
    plan.transform_2d(f.re.data(), f.im.data(), f.scratch.data(), FftSign::Inverse);

    float worst = 0.0f, scale = 0.0f;
    for (std::size_t i = 0; i < f.im.size(); ++i) {
        worst = std::fmax(worst, std::fabs(f.im[i]));
        scale = std::fmax(scale, std::fabs(f.re[i]));
    }
    CAPTURE(worst);
    CAPTURE(scale);
    CHECK(worst / scale < 1e-5f);
}

TEST_CASE("the transform is linear")
{
    constexpr std::uint32_t n = 64;
    FftPlan plan{n};
    std::vector<float> scratch(4 * n);

    std::vector<float> ar(n), ai(n), br(n), bi(n);
    fill_random(ar, ai, 11u);
    fill_random(br, bi, 22u);

    std::vector<float> sr(n), si(n);
    for (std::uint32_t j = 0; j < n; ++j) {
        sr[j] = ar[j] + 3.0f * br[j];
        si[j] = ai[j] + 3.0f * bi[j];
    }

    plan.transform(ar.data(), ai.data(), 1, scratch.data(), FftSign::Forward);
    plan.transform(br.data(), bi.data(), 1, scratch.data(), FftSign::Forward);
    plan.transform(sr.data(), si.data(), 1, scratch.data(), FftSign::Forward);

    for (std::uint32_t k = 0; k < n; ++k) {
        CHECK(std::abs(sr[k] - (ar[k] + 3.0f * br[k])) < 1e-3f);
        CHECK(std::abs(si[k] - (ai[k] + 3.0f * bi[k])) < 1e-3f);
    }
}

TEST_CASE("strided transforms agree bit-for-bit with contiguous ones")
{
    // The column pass of the 2D transform is a strided 1D transform; this
    // isolates that path from the rest of the 2D machinery. Because the strided
    // case gathers into the same contiguous scratch and runs the same butterfly
    // code, the results must be identical, not merely close.
    constexpr std::uint32_t n = 64;
    FftPlan plan{n};
    std::vector<float> scratch(4 * n);

    std::vector<float> re(n), im(n);
    fill_random(re, im, 9090u);

    std::vector<float> flat_re = re, flat_im = im;
    plan.transform(flat_re.data(), flat_im.data(), 1, scratch.data(), FftSign::Forward);

    constexpr std::ptrdiff_t stride = 5;
    std::vector<float> s_re(n * stride, 0.0f), s_im(n * stride, 0.0f);
    for (std::uint32_t j = 0; j < n; ++j) {
        s_re[j * stride] = re[j];
        s_im[j * stride] = im[j];
    }
    plan.transform(s_re.data(), s_im.data(), stride, scratch.data(), FftSign::Forward);

    for (std::uint32_t j = 0; j < n; ++j) {
        CHECK(s_re[j * stride] == flat_re[j]);
        CHECK(s_im[j * stride] == flat_im[j]);
    }
}

TEST_CASE("row and column ranges compose into the full 2D transform")
{
    // Guards the decomposition the threaded scheduler will rely on: splitting
    // the index range must not change the answer by a single bit, and the
    // ranges here are deliberately not aligned to any power of two.
    constexpr std::uint32_t n = 32;
    FftPlan plan{n};

    Field whole{n}, split{n};
    fill_random(whole.re, whole.im, 5150u);
    split.re = whole.re;
    split.im = whole.im;

    plan.transform_2d(whole.re.data(), whole.im.data(), whole.scratch.data(),
                      FftSign::Inverse);

    for (std::uint32_t r = 0; r < n; r += 7) {
        plan.transform_row_range(split.re.data(), split.im.data(), r,
                                 std::min(r + 7, n), split.scratch.data(),
                                 FftSign::Inverse);
    }
    for (std::uint32_t c = 0; c < n; c += 5) {
        plan.transform_col_range(split.re.data(), split.im.data(), c,
                                 std::min(c + 5, n), split.scratch.data(),
                                 FftSign::Inverse);
    }

    CHECK(std::memcmp(whole.re.data(), split.re.data(),
                      whole.re.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(whole.im.data(), split.im.data(),
                      whole.im.size() * sizeof(float)) == 0);
}

TEST_CASE("plan rejects sizes that are not powers of two")
{
    CHECK_THROWS_AS(FftPlan{12}, std::invalid_argument);
    CHECK_THROWS_AS(FftPlan{0}, std::invalid_argument);
    CHECK_NOTHROW(FftPlan{1});
    CHECK_NOTHROW(FftPlan{1024});
}

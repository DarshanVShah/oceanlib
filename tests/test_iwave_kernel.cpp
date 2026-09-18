// The iWave convolution kernel.
//
// The kernel IS the physics: its Fourier symbol is the dispersion relation the
// interaction field will reproduce. So these tests check the symbol directly,
// analytically, rather than inferring it from a simulation - a ripple-
// expansion test (test_interaction.cpp) then confirms end to end that the
// solver really delivers what the symbol promises.
#include "core/iwave_kernel.hpp"

#include <doctest/doctest.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace ocean::detail;

namespace {
constexpr double kPi = 3.14159265358979323846;
}

TEST_CASE("bessel_j0 matches known values and zeros")
{
    CHECK(bessel_j0(0.0) == doctest::Approx(1.0).epsilon(1e-7));

    // J0 is even.
    CHECK(bessel_j0(-1.7) == doctest::Approx(bessel_j0(1.7)).epsilon(1e-12));

    // The first five zeros of J0, to 12 digits. Hitting these pins both
    // branches of the A&S approximation (the first is below the x = 3
    // crossover, the rest above it) and would catch a transcription slip in
    // any coefficient.
    const double zeros[5] = {2.404825557695773, 5.520078110286311,
                             8.653727912911012, 11.791534439014281,
                             14.930917708487785};
    for (double z : zeros) {
        CHECK(std::fabs(bessel_j0(z)) < 2e-7);
    }

    // A couple of tabulated interior values, so the test is not satisfied by a
    // function that merely has the right roots.
    CHECK(bessel_j0(1.0) == doctest::Approx(0.7651976866).epsilon(1e-6));
    CHECK(bessel_j0(5.0) == doctest::Approx(-0.1775967713).epsilon(1e-6));
    CHECK(bessel_j0(10.0) == doctest::Approx(-0.2459357645).epsilon(1e-6));
}

TEST_CASE("kernel is exactly 8-fold symmetric")
{
    // Not "symmetric to within a tolerance" - bit-for-bit equal. Every orbit
    // member is written from one stored value, so any asymmetry would be a
    // structural bug rather than accumulated rounding, and an exact check is
    // the one that catches it.
    for (auto method : {KernelMethod::Hankel, KernelMethod::LeastSquares}) {
        for (std::uint32_t p : {3u, 4u, 6u, 8u}) {
            IWaveKernel k;
            build_iwave_kernel(p, 0.25, 0.0, method, k);
            const int P = static_cast<int>(p);
            for (int j = -P; j <= P; ++j) {
                for (int i = -P; i <= P; ++i) {
                    const float v = k.tap(i, j);
                    REQUIRE(k.tap(-i,  j) == v);   // mirror in x
                    REQUIRE(k.tap( i, -j) == v);   // mirror in z
                    REQUIRE(k.tap( j,  i) == v);   // transpose
                    REQUIRE(k.tap(-j, -i) == v);   // anti-transpose
                }
            }
        }
    }
}

TEST_CASE("kernel sums to zero, so a uniform water level feels no force")
{
    // The true symbol is exactly 0 at k = 0. A kernel with a non-zero sum
    // would give the DC mode a restoring force - and because the operator
    // cannot propagate DC anywhere, the result would be a constant offset
    // oscillating in place forever. This is the reason for the constraint,
    // not neatness.
    for (std::uint32_t p : {3u, 4u, 6u, 8u}) {
        IWaveKernel lsq;
        build_iwave_kernel(p, 0.25, 0.0, KernelMethod::LeastSquares, lsq);

        // Enforced by elimination, so it holds to float rounding of the sum
        // itself, not to fit tolerance.
        double sum = 0.0;
        for (std::size_t t = 0; t < lsq.taps.size(); ++t) sum += lsq.taps[t];
        const double scale = std::fabs(static_cast<double>(lsq.tap(0, 0)));
        CHECK(std::fabs(sum) < 1e-5 * scale);

        // And therefore the realised symbol vanishes at DC.
        CHECK(std::fabs(realised_symbol(lsq, 0.0, 0.0)) < 1e-5 * scale);

        IWaveKernel hk;
        build_iwave_kernel(p, 0.25, 0.0, KernelMethod::Hankel, hk);
        double hsum = 0.0;
        for (std::size_t t = 0; t < hk.taps.size(); ++t) hsum += hk.taps[t];
        const double hscale = std::fabs(static_cast<double>(hk.tap(0, 0)));
        CHECK(std::fabs(hsum) < 1e-5 * hscale);
    }
}

TEST_CASE("deep-water kernel scales exactly as 1/dx")
{
    // With S(k) = k the fit is scale-free: substituting kappa = k*dx makes
    // every dx cancel out of the weighted normal equations, so the stencil for
    // cell size dx must be the stencil for cell size 1 divided by dx. This is
    // a real structural property of the derivation, and it catches a dx that
    // was applied in one place but not another - the classic units bug.
    IWaveKernel a, b;
    build_iwave_kernel(6, 0.25, 0.0, KernelMethod::LeastSquares, a);
    build_iwave_kernel(6, 0.50, 0.0, KernelMethod::LeastSquares, b);

    for (int j = -6; j <= 6; ++j) {
        for (int i = -6; i <= 6; ++i) {
            CHECK(static_cast<double>(b.tap(i, j)) ==
                  doctest::Approx(0.5 * static_cast<double>(a.tap(i, j)))
                      .epsilon(1e-4));
        }
    }
}

TEST_CASE("realised symbol tracks the true dispersion relation")
{
    // THE load-bearing test for the method's physics: the kernel's Fourier
    // symbol IS the dispersion relation, so its error IS the dispersion error.
    // Measured and reported rather than merely bounded, because the number is
    // what justifies a particular kernel_radius.
    //
    // Reported over [band_low, k_max], the band the kernel claims. Note the
    // wave-SPEED error is half the symbol error, since omega = sqrt(g*S).
    const double cell  = 0.25;
    const double k_max = kPi / cell;

    std::printf("\n  iWave kernel: peak |realised/true - 1| over the claimed "
                "band, deep water, dx = %.2f m\n", cell);
    std::printf("  (band low edge = %.3g * k_max, i.e. wavelengths from 2 to "
                "%.0f cells)\n", 0.125, 2.0 / 0.125);
    std::printf("  %-3s %-6s %10s %10s   %-14s %s\n", "P", "taps", "Hankel",
                "LeastSq", "winner", "LSQ speed err");

    for (std::uint32_t p : {3u, 4u, 6u, 8u, 10u}) {
        IWaveKernel hk, lq;
        build_iwave_kernel(p, cell, 0.0, KernelMethod::Hankel, hk);
        build_iwave_kernel(p, cell, 0.0, KernelMethod::LeastSquares, lq);
        const unsigned taps = (2u * p + 1u) * (2u * p + 1u);
        std::printf("  %-3u %-6u %9.3f%% %9.3f%%   %-14s %8.3f%%\n", p, taps,
                    100.0 * hk.peak_relative_error,
                    100.0 * lq.peak_relative_error,
                    lq.peak_relative_error < hk.peak_relative_error
                        ? "least-squares" : "Hankel",
                    50.0 * lq.peak_relative_error);

        // Both derivations must be recognisably approximating |k|. An order of
        // magnitude is the line between "truncation error" and "the derivation
        // is wrong" - which is exactly what caught the missing cell-area factor
        // in the Hankel transform, a constant 1/dx^2 that survived every P.
        CHECK(hk.peak_relative_error < 0.75);
        CHECK(lq.peak_relative_error < 0.60);

        // The band-restricted least-squares fit must beat the truncated
        // transform inside the band both are measured over. This is the whole
        // justification for not simply shipping Tessendorf's kernel, so it is
        // asserted, not just printed.
        if (p >= 4) CHECK(lq.peak_relative_error < hk.peak_relative_error);

        // The shipped configuration, pinned to what is actually measured so a
        // regression cannot pass quietly. P=6 is the default (the cost knee,
        // and Tessendorf's own choice); P=8 roughly halves the error for 1.7x
        // the taps, which is the trade a caller is buying when they raise it.
        if (p == 6)  CHECK(lq.peak_relative_error < 0.15);   // measured 13.1%
        if (p == 8)  CHECK(lq.peak_relative_error < 0.08);   // measured  6.5%
        if (p == 10) CHECK(lq.peak_relative_error < 0.04);   // measured  3.0%

        // The symbol must be non-negative everywhere the grid can represent -
        // corners of the square included, not just the inscribed circle. A
        // negative realised symbol means omega^2 = g*S < 0 for that mode: an
        // exponentially growing solution that no timestep can rescue, and one
        // that would sit in the aliased corner where nobody thinks to look.
        CHECK(lq.max_symbol > 0.0);
        for (int zi = 0; zi <= 96; ++zi) {
            for (int xi = 0; xi <= 96; ++xi) {
                const double kx = k_max * xi / 96.0;
                const double kz = k_max * zi / 96.0;
                // LSQ only: the truncated-transform kernel is knowingly
                // negative in the aliased corners, which is measured and
                // asserted in its own test rather than treated as a surprise
                // here.
                REQUIRE(realised_symbol(lq, kx, kz) > -1e-3 * lq.max_symbol);
            }
        }
    }
}

TEST_CASE("larger kernel radius widens the usable band")
{
    // The accuracy/cost trade-off, as a property rather than a table: paying
    // for more taps must buy a monotonically better symbol. If it ever did
    // not, the extra work would be bought for nothing and the radius knob
    // would be meaningless.
    double prev = 1e9;
    for (std::uint32_t p : {3u, 4u, 6u, 8u, 10u}) {
        IWaveKernel k;
        build_iwave_kernel(p, 0.25, 0.0, KernelMethod::LeastSquares, k);
        CHECK(k.peak_relative_error < prev);
        prev = k.peak_relative_error;
    }
}

TEST_CASE("dispersion error profile across the band")
{
    // One number hides where the error actually is, and for this operator the
    // answer is strongly k-dependent: a finite even stencil cannot match |k|
    // near DC no matter how large P grows, while at high k the truncation is
    // what limits it. Printing the profile is what turns "the kernel is
    // approximate" into a statement about which wavelengths are trustworthy.
    const double cell  = 0.25;
    const double k_max = kPi / cell;
    const double frac[] = {1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4,
                           1.0 / 2, 3.0 / 4, 1.0};

    std::printf("\n  realised/true symbol along the kx axis, dx = %.2f m "
                "(k_max = %.2f rad/m)\n", cell, k_max);
    std::printf("  %-14s", "k / k_max:");
    for (double f : frac) std::printf(" %8.4f", f);
    std::printf("\n  %-14s", "wavelength m:");
    for (double f : frac) std::printf(" %8.2f", 2.0 * kPi / (k_max * f));
    std::printf("\n");

    for (std::uint32_t p : {4u, 6u, 8u}) {
        for (auto method : {KernelMethod::Hankel, KernelMethod::LeastSquares}) {
            IWaveKernel k;
            build_iwave_kernel(p, cell, 0.0, method, k);
            char label[32];
            std::snprintf(label, sizeof(label), "P=%u %s", p,
                          method == KernelMethod::Hankel ? "Hankel" : "LSQ");
            std::printf("  %-14s", label);
            for (double f : frac) {
                const double kk = k_max * f;
                std::printf(" %8.4f", realised_symbol(k, kk, 0.0) /
                                      iwave_symbol(kk, 0.0));
            }
            std::printf("\n");
        }
    }
    // How sensitive is the fit to where its band starts? Narrowing the band
    // buys accuracy inside it and gives up everything below - so the default
    // is a choice about which wavelengths this field claims, and this is the
    // measurement that makes it a choice rather than a guess.
    std::printf("\n  LSQ band-start sweep, P = 6 (worst |ratio-1| over the band)\n");
    for (double bf : {0.0625, 0.125, 0.25, 0.375}) {
        IWaveKernel k;
        build_iwave_kernel(6, cell, 0.0, KernelMethod::LeastSquares, k, 96, bf);
        char label[32];
        std::snprintf(label, sizeof(label), "band>=%.4g", bf);
        std::printf("  %-14s", label);
        double worst = 0.0;
        for (double f : frac) {
            const double kk = k_max * f;
            const double r = realised_symbol(k, kk, 0.0) / iwave_symbol(kk, 0.0);
            std::printf(" %8.4f", r);
            if (f >= bf) worst = std::fmax(worst, std::fabs(r - 1.0));
        }
        std::printf("   worst in band %.2f%%\n", 100.0 * worst);
    }

    // Diagonal direction too: a radially symmetric operator must not favour
    // the grid axes, and an isotropy failure shows up here and nowhere else.
    {
        IWaveKernel k;
        build_iwave_kernel(6, cell, 0.0, KernelMethod::LeastSquares, k);
        std::printf("  %-14s", "P=6 LSQ diag");
        for (double f : frac) {
            const double kk = k_max * f;
            const double c  = kk / std::sqrt(2.0);
            std::printf(" %8.4f", realised_symbol(k, c, c) /
                                  iwave_symbol(kk, 0.0));
        }
        std::printf("\n");
    }
}

TEST_CASE("low-wavenumber breakdown is located, not hidden")
{
    // A finite even stencil has an analytic, even symbol, so near k = 0 it
    // behaves like c*k^2 while the true symbol behaves like k. The relative
    // error therefore diverges as k -> 0 for ANY finite kernel. That is a real
    // limit of the method, so this test finds where it bites rather than
    // pretending it does not exist: below this wavenumber, long waves are the
    // FFT ocean's job and the interaction field must not be trusted.
    const double cell  = 0.25;
    const double k_max = kPi / cell;
    IWaveKernel k;
    build_iwave_kernel(6, cell, 0.0, KernelMethod::LeastSquares, k);

    double k_break = 0.0;
    for (int i = 1; i <= 2000; ++i) {
        const double kk = k_max * i / 2000.0;
        const double e  = std::fabs(realised_symbol(k, kk, 0.0) /
                                    iwave_symbol(kk, 0.0) - 1.0);
        if (e < 0.10) { k_break = kk; break; }
    }
    REQUIRE(k_break > 0.0);
    std::printf("  P=6, dx=%.2f m: dispersion within 10%% for wavelengths "
                "below %.2f m (k >= %.2f rad/m)\n",
                cell, 2.0 * kPi / k_break, k_break);

    // It must break down at LONG waves, not short ones. If this ever inverted,
    // the kernel would be approximating something other than |k|.
    CHECK(k_break < 0.5 * k_max);
}

TEST_CASE("finite depth changes the kernel, and reduces to deep water")
{
    // Same relationship ADR-019 established for the FFT ocean: depth enters
    // only through the dispersion relation. A shallow interaction field and a
    // shallow FFT ocean therefore stay consistent for free.
    IWaveKernel deep, shallow, very_deep;
    build_iwave_kernel(6, 0.25, 0.0,    KernelMethod::LeastSquares, deep);
    build_iwave_kernel(6, 0.25, 0.30,   KernelMethod::LeastSquares, shallow);
    build_iwave_kernel(6, 0.25, 1.0e6,  KernelMethod::LeastSquares, very_deep);

    // tanh(k*h) <= 1 always, so a shallow symbol can only be lower than the
    // deep one at the same k - never higher. A sign error in the tanh argument
    // would violate this immediately.
    for (int i = 1; i <= 64; ++i) {
        const double kk = (kPi / 0.25) * i / 64.0;
        CHECK(iwave_symbol(kk, 0.30) <= iwave_symbol(kk, 0.0) + 1e-12);
    }

    bool differs = false;
    for (int j = -6; j <= 6 && !differs; ++j) {
        for (int i = -6; i <= 6; ++i) {
            if (std::fabs(deep.tap(i, j) - shallow.tap(i, j)) >
                1e-3 * std::fabs(deep.tap(0, 0))) { differs = true; break; }
        }
    }
    CHECK(differs);

    // An enormous finite depth must converge back onto the deep branch, even
    // though the deep branch is taken as a literal special case rather than
    // computed as a limit.
    for (int j = -6; j <= 6; ++j) {
        for (int i = -6; i <= 6; ++i) {
            CHECK(static_cast<double>(very_deep.tap(i, j)) ==
                  doctest::Approx(static_cast<double>(deep.tap(i, j)))
                      .epsilon(1e-5));
        }
    }
}

TEST_CASE("kernel rejects out-of-range configuration")
{
    IWaveKernel k;
    CHECK_THROWS_AS(build_iwave_kernel(0, 0.25, 0.0, KernelMethod::LeastSquares, k),
                    std::invalid_argument);
    CHECK_THROWS_AS(build_iwave_kernel(13, 0.25, 0.0, KernelMethod::LeastSquares, k),
                    std::invalid_argument);
    CHECK_THROWS_AS(build_iwave_kernel(6, 0.0, 0.0, KernelMethod::LeastSquares, k),
                    std::invalid_argument);
    CHECK_THROWS_AS(build_iwave_kernel(6, -1.0, 0.0, KernelMethod::LeastSquares, k),
                    std::invalid_argument);
}

TEST_CASE("the truncated-transform kernel is unstable in the aliased corners")
{
    // A measured result about the METHOD, not about one configuration, and the
    // single strongest reason this library does not simply ship Tessendorf's
    // kernel.
    //
    // Truncating the inverse Hankel transform gives a kernel whose realised
    // symbol is faithful throughout the inscribed Nyquist disc - the band of
    // genuinely resolvable waves - but which dips to roughly -0.12 to -0.18 of
    // its own maximum in the CORNERS of k-space, beyond that disc, at every
    // radius from 2 to 12. Those corner modes have wavelengths shorter than
    // two cells along the diagonal, so they are aliasing artefacts rather than
    // waves; but the grid can still hold them, and omega^2 = g*S with S < 0
    // means they grow exponentially instead of oscillating.
    //
    // Nothing deliberately excites a checkerboard, which is exactly why this
    // is dangerous: rounding noise and sharply-injected sources excite it a
    // little, damping does not remove an exponential, and it surfaces much
    // later as "the water sometimes explodes".
    //
    // Fitting the kernel over a band instead removes it for free: the fit
    // includes the corners at low weight, which is enough to hold the symbol
    // non-negative there without spending accuracy in the band that matters.
    const double cell  = 0.25;
    const double k_max = kPi / cell;

    auto scan = [&](const IWaveKernel& k, bool disc_only) {
        double lo = 0.0, hi = 1e-30;
        for (int zi = 0; zi <= 160; ++zi) {
            for (int xi = 0; xi <= 160; ++xi) {
                const double kx = k_max * xi / 160.0;
                const double kz = k_max * zi / 160.0;
                if (disc_only && std::sqrt(kx * kx + kz * kz) > k_max) continue;
                const double rs = realised_symbol(k, kx, kz);
                hi = std::fmax(hi, rs);
                lo = std::fmin(lo, rs);
            }
        }
        return lo / hi;
    };

    std::printf("\n  realised symbol minimum as a fraction of its maximum\n");
    std::printf("  %-16s %12s %14s\n", "kernel", "in disc", "incl corners");

    for (std::uint32_t p : {3u, 4u, 6u, 8u, 10u}) {
        IWaveKernel hk, lq;
        build_iwave_kernel(p, cell, 0.0, KernelMethod::Hankel,       hk);
        build_iwave_kernel(p, cell, 0.0, KernelMethod::LeastSquares, lq);

        char lh[32], ll[32];
        std::snprintf(lh, sizeof(lh), "P=%u Hankel", p);
        std::snprintf(ll, sizeof(ll), "P=%u LSQ", p);
        const double h_disc = scan(hk, true), h_all = scan(hk, false);
        const double l_disc = scan(lq, true), l_all = scan(lq, false);
        std::printf("  %-16s %11.5f %13.5f\n", lh, h_disc, h_all);
        std::printf("  %-16s %11.5f %13.5f\n", ll, l_disc, l_all);

        // Both are sound inside the disc: the truncated transform is a correct
        // derivation for genuinely resolvable waves, and saying otherwise
        // would misrepresent it.
        CHECK(h_disc > -1e-3);
        CHECK(l_disc > -1e-3);

        // The difference is entirely in the corners.
        CHECK(h_all < -0.05);            // unstable, at every radius
        CHECK(l_all > -1e-3);            // sound everywhere
        CHECK_FALSE(kernel_is_stable(hk));
        CHECK(kernel_is_stable(lq));
    }

    // At every radius, without exception - so this is a property of the
    // derivation, not of a particular truncation length.
    for (std::uint32_t p : {2u, 3u, 4u, 6u, 8u, 10u, 12u}) {
        IWaveKernel hk;
        build_iwave_kernel(p, cell, 0.0, KernelMethod::Hankel, hk);
        CHECK_FALSE(kernel_is_stable(hk));
    }

    // The shipped derivation survives everywhere, across cell sizes spanning
    // three orders of magnitude - confirming this is a property of the method
    // rather than of one lucky configuration. (Deep water makes the kernel
    // exactly scale-free, so these must agree; a failure here would mean the
    // scale-freeness proved in the 1/dx test had been broken elsewhere.)
    for (double c : {0.02, 0.25, 2.0}) {
        for (std::uint32_t p : {3u, 4u, 6u, 8u, 10u, 12u}) {
            IWaveKernel ok;
            build_iwave_kernel(p, c, 0.0, KernelMethod::LeastSquares, ok);
            CHECK(kernel_is_stable(ok));
        }
    }
}

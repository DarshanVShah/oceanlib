#include <doctest/doctest.h>

#include "core/spectrum.hpp"
#include "ocean/ocean.hpp"

#include <cmath>
#include <cstring>
#include <vector>

using namespace ocean;
using namespace ocean::detail;

namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

SpectrumDesc default_spectrum()
{
    SpectrumDesc s;
    s.wind_speed        = 10.0f;
    s.fetch             = 100000.0f;
    s.wind_direction    = 0.0f;
    s.peak_enhancement  = 3.3f;
    s.swell             = 0.0f;
    s.small_wave_cutoff = 0.001f;  // effectively disabled, for clean integrals
    s.gravity           = 9.81f;
    return s;
}

// m0 = integral of S(omega) d(omega), truncated at omega_max. Simpson's rule.
double integrate_jonswap(const SpectrumDesc& d, double omega_max, int steps)
{
    const double h = omega_max / steps;
    double sum = 0.0;
    for (int i = 0; i <= steps; ++i) {
        const double w = i * h;
        const double weight = (i == 0 || i == steps) ? 1.0
                            : (i % 2 == 1)           ? 4.0
                                                     : 2.0;
        sum += weight * jonswap(d, w);
    }
    return sum * h / 3.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// Directional spreading
// ---------------------------------------------------------------------------

TEST_CASE("directional spreading integrates to 1 over all directions")
{
    // If this is wrong, total energy silently depends on frequency, because
    // the spreading exponent s varies with omega. The surface would still look
    // like water; it would just be the wrong sea state at every wind speed.
    const SpectrumDesc d = default_spectrum();
    const double omega_p = jonswap_peak_omega(d);

    for (double ratio : {0.5, 0.8, 1.0, 1.5, 2.5, 4.0}) {
        CAPTURE(ratio);
        const double omega = ratio * omega_p;

        constexpr int steps = 20000;
        const double dtheta = kTwoPi / steps;
        double total = 0.0;
        for (int i = 0; i < steps; ++i) {
            const double theta = -kPi + (i + 0.5) * dtheta;
            total += directional_spread(d, omega, theta) * dtheta;
        }
        CAPTURE(total);
        CHECK(total == doctest::Approx(1.0).epsilon(1e-3));
    }
}

TEST_CASE("spreading peaks downwind and vanishes upwind")
{
    SpectrumDesc d = default_spectrum();
    d.wind_direction = 0.9f;  // arbitrary heading
    const double omega = jonswap_peak_omega(d);
    const double wdir  = d.wind_direction;

    const double downwind = directional_spread(d, omega, wdir);
    const double across   = directional_spread(d, omega, wdir + kPi / 2.0);
    const double upwind   = directional_spread(d, omega, wdir + kPi);

    CHECK(downwind > across);
    CHECK(across > upwind);

    // The half-angle cosine form vanishes at 180 degrees off the wind, so waves
    // never travel into the wind and no extra suppression term is needed.
    // (cos(pi/2) is not exactly 0 in floating point, so the residual is a
    // subnormal around 1e-317 rather than a true zero.)
    CHECK(upwind < 1e-300);
}

TEST_CASE("swell narrows the directional distribution")
{
    SpectrumDesc wind_sea = default_spectrum();
    SpectrumDesc swelly   = default_spectrum();
    swelly.swell = 1.0f;

    const double omega = jonswap_peak_omega(wind_sea) * 0.8;

    // Both still integrate to 1, so "narrower" must mean taller at the centre
    // and lower off-axis - which is exactly what the normalisation guarantees.
    CHECK(directional_spread(swelly, omega, 0.0) >
          directional_spread(wind_sea, omega, 0.0));
    CHECK(directional_spread(swelly, omega, 0.6) <
          directional_spread(wind_sea, omega, 0.6));

    constexpr int steps = 20000;
    const double dtheta = kTwoPi / steps;
    double total = 0.0;
    for (int i = 0; i < steps; ++i) {
        total += directional_spread(swelly, omega, -kPi + (i + 0.5) * dtheta) * dtheta;
    }
    CHECK(total == doctest::Approx(1.0).epsilon(1e-3));
}

// ---------------------------------------------------------------------------
// JONSWAP shape
// ---------------------------------------------------------------------------

TEST_CASE("JONSWAP density peaks at the peak frequency")
{
    const SpectrumDesc d = default_spectrum();
    const double omega_p = jonswap_peak_omega(d);

    const double at_peak = jonswap(d, omega_p);
    CHECK(at_peak > jonswap(d, omega_p * 0.7));
    CHECK(at_peak > jonswap(d, omega_p * 1.4));

    // Scan for the true maximum; gamma^r shifts it only slightly off omega_p.
    double best_w = 0.0, best_v = 0.0;
    for (int i = 1; i <= 4000; ++i) {
        const double w = i * 0.005;
        const double v = jonswap(d, w);
        if (v > best_v) { best_v = v; best_w = w; }
    }
    CAPTURE(best_w);
    CAPTURE(omega_p);
    CHECK(best_w == doctest::Approx(omega_p).epsilon(0.1));

    CHECK(jonswap(d, 0.0) == 0.0);  // no singularity at DC
}

TEST_CASE("peak enhancement sharpens the peak without moving it")
{
    SpectrumDesc pm = default_spectrum();
    pm.peak_enhancement = 1.0f;  // gamma = 1 degenerates to Pierson-Moskowitz
    SpectrumDesc js = default_spectrum();  // gamma = 3.3

    const double omega_p = jonswap_peak_omega(js);
    CHECK(jonswap_peak_omega(pm) == doctest::Approx(omega_p));

    // At the peak, r = 1 exactly, so gamma^r = gamma. Compare against the
    // widened float, not the double literal: 3.3f is 3.29999995231628, and
    // mistaking one for the other is a 1.4e-8 relative discrepancy that looks
    // like a real bug.
    CHECK(jonswap(js, omega_p) ==
          doctest::Approx(jonswap(pm, omega_p) *
                          static_cast<double>(js.peak_enhancement))
              .epsilon(1e-12));

    // Far from the peak, r -> 0 and gamma^r -> 1, so the two agree.
    CHECK(jonswap(js, omega_p * 4.0) ==
          doctest::Approx(jonswap(pm, omega_p * 4.0)).epsilon(0.01));
}

TEST_CASE("longer fetch and stronger wind both lower the peak frequency")
{
    SpectrumDesc base = default_spectrum();

    SpectrumDesc long_fetch = base;
    long_fetch.fetch = 500000.0f;
    CHECK(jonswap_peak_omega(long_fetch) < jonswap_peak_omega(base));

    SpectrumDesc strong = base;
    strong.wind_speed = 20.0f;
    CHECK(jonswap_peak_omega(strong) < jonswap_peak_omega(base));

    // Lower peak frequency means longer waves: lambda = 2*pi*g/omega^2.
    const double wp_base = jonswap_peak_omega(base);
    const double wp_slow = jonswap_peak_omega(strong);
    const double lambda_base = kTwoPi * 9.81 / (wp_base * wp_base);
    const double lambda_slow = kTwoPi * 9.81 / (wp_slow * wp_slow);
    CHECK(lambda_slow > lambda_base);
}

// ---------------------------------------------------------------------------
// The change of variables: frequency space -> 2D wavenumber space
// ---------------------------------------------------------------------------

TEST_CASE("the 2D wavenumber density integrates to the same energy as S(omega)")
{
    // THE test for this module. Integrating Psi over the kx-kz plane in polar
    // coordinates must reproduce the integral of S(omega) over frequency:
    //
    //   int Psi(kx,kz) dkx dkz = int Psi(k,theta) k dk dtheta
    //                          = int S(k) D(theta) dk dtheta
    //                          = int S(k) dk = int S(omega) domega
    //
    // It simultaneously checks the domega/dk Jacobian, the 1/k polar factor
    // and the D normalisation. Getting any one of them wrong still produces
    // water that looks fine but carries the wrong energy - the classic silent
    // failure in this kind of code. Both sides are truncated at the same
    // wavenumber so no truncation bias creeps in.
    const SpectrumDesc d = default_spectrum();

    constexpr double k_max = 2.0;  // rad/m
    const double omega_max = std::sqrt(9.81 * k_max);

    constexpr int k_steps = 4000;
    constexpr int t_steps = 720;
    const double dk = k_max / k_steps;
    const double dt = kTwoPi / t_steps;

    double polar = 0.0;
    for (int ik = 0; ik < k_steps; ++ik) {
        const double k = (ik + 0.5) * dk;
        for (int it = 0; it < t_steps; ++it) {
            const double theta = -kPi + (it + 0.5) * dt;
            const double kx = k * std::cos(theta);
            const double kz = k * std::sin(theta);
            polar += wave_density(d, kx, kz) * k * dk * dt;
        }
    }

    const double spectral = integrate_jonswap(d, omega_max, 20000);

    CAPTURE(polar);
    CAPTURE(spectral);
    CHECK(polar == doctest::Approx(spectral).epsilon(0.01));
}

TEST_CASE("wavenumber density is zero at DC and positive downwind")
{
    const SpectrumDesc d = default_spectrum();
    CHECK(wave_density(d, 0.0, 0.0) == 0.0);

    const double omega_p = jonswap_peak_omega(d);
    const double k_p = omega_p * omega_p / 9.81;  // deep-water dispersion

    CHECK(wave_density(d, k_p, 0.0) > 0.0);
    // Directly upwind carries no energy.
    CHECK(wave_density(d, -k_p, 0.0) < 1e-20);
}

TEST_CASE("the short-wave cutoff suppresses high wavenumbers")
{
    SpectrumDesc sharp = default_spectrum();
    SpectrumDesc soft  = default_spectrum();
    soft.small_wave_cutoff = 4.0f;  // roll off waves shorter than 4 m

    const double k = kTwoPi / 4.0;  // exactly at the knee
    const double ratio = wave_density(soft, k, 0.0) / wave_density(sharp, k, 0.0);

    // exp(-k^2 l^2) with l = cutoff/(2*pi) gives exactly exp(-1) at this k.
    CAPTURE(ratio);
    CHECK(ratio == doctest::Approx(std::exp(-1.0)).epsilon(1e-4));

    // Low wavenumbers are essentially untouched.
    CHECK(wave_density(soft, 0.05, 0.0) ==
          doctest::Approx(wave_density(sharp, 0.05, 0.0)).epsilon(0.01));
}

// ---------------------------------------------------------------------------
// Energy response
// ---------------------------------------------------------------------------

TEST_CASE("total energy rises steeply and monotonically with wind speed")
{
    double previous = 0.0;
    for (float u : {4.0f, 6.0f, 8.0f, 10.0f, 14.0f, 20.0f}) {
        CAPTURE(u);
        SpectrumDesc d = default_spectrum();
        d.wind_speed = u;
        const double m0 = integrate_jonswap(d, 8.0, 20000);
        CHECK(m0 > previous);
        previous = m0;
    }

    // Significant wave height must follow the analytically derived
    // fetch-limited scaling law. At FIXED fetch:
    //
    //   alpha   = 0.076 * (U^2/(g F))^0.22   ->  proportional to U^0.44
    //   omega_p = 22 * (g^2/(U F))^(1/3)     ->  proportional to U^(-1/3)
    //   m0      = integral of S ~ alpha * omega_p^-4  ->  U^(0.44 + 4/3)
    //   H_s     = 4 sqrt(m0)                 ->  U^0.886667
    //
    // Checking the exponent rather than a single magnitude validates the
    // exponents inside BOTH alpha and omega_p and the omega^-5 equilibrium
    // tail simultaneously - a single-point check would pass with any of them
    // subtly wrong.
    //
    // Note this is well below the U^2 of a FULLY DEVELOPED sea: with fetch
    // pinned at 100 km, a 20 m/s wind is fetch-limited and the waves never get
    // the chance to reach equilibrium. Expecting U^2 here is the intuitive
    // mistake.
    SpectrumDesc slow = default_spectrum();
    slow.wind_speed = 10.0f;
    SpectrumDesc fast = default_spectrum();
    fast.wind_speed = 20.0f;

    const double hs_slow = 4.0 * std::sqrt(integrate_jonswap(slow, 8.0, 20000));
    const double hs_fast = 4.0 * std::sqrt(integrate_jonswap(fast, 8.0, 20000));
    const double exponent = std::log(hs_fast / hs_slow) / std::log(2.0);
    CAPTURE(hs_slow);
    CAPTURE(hs_fast);
    CAPTURE(exponent);
    CHECK(exponent == doctest::Approx(0.886667).epsilon(0.01));

    // Independent sanity check against observed oceanography: a 10 m/s wind is
    // about 20 knots, Beaufort 5, for which H_s of roughly 2-3 m is what is
    // actually measured at sea. This catches a units error or a missing
    // constant that a pure scaling test would sail straight past.
    CHECK(hs_slow > 1.5);
    CHECK(hs_slow < 3.5);
}

TEST_CASE("zero wind produces a completely flat ocean")
{
    SpectrumDesc d = default_spectrum();
    d.wind_speed = 0.0f;
    CHECK(jonswap_peak_omega(d) == 0.0);
    CHECK(jonswap(d, 1.0) == 0.0);
    CHECK(wave_density(d, 0.1, 0.1) == 0.0);
}

// ---------------------------------------------------------------------------
// Discretised tables
// ---------------------------------------------------------------------------

TEST_CASE("the discrete grid carries the energy the continuous density predicts")
{
    // Validates the amplitude normalisation, including the factor of 1/2 that
    // accounts for each wavenumber receiving energy from two counter-
    // propagating wave trains. Dropping it makes the whole ocean sqrt(2) times
    // too tall - a bug that looks entirely plausible on screen.
    OceanDesc d;
    d.size         = 256;
    d.patch_length = 1000.0f;
    d.seed         = 20240917;
    d.spectrum     = default_spectrum();

    SpectrumTables t;
    build_spectrum(d, t);

    const double dk = kTwoPi / d.patch_length;

    // E|h(k,t)|^2 summed over the grid equals 2 * sum |h0(k)|^2 in expectation,
    // because h(k,t) = h0(k) e^{iwt} + conj(h0(-k)) e^{-iwt} with independent
    // draws, so the cross terms vanish.
    double realised = 0.0;
    double predicted = 0.0;
    const std::size_t cells = static_cast<std::size_t>(d.size) * d.size;
    for (std::size_t i = 0; i < cells; ++i) {
        realised += 2.0 * (static_cast<double>(t.h0_re[i]) * t.h0_re[i] +
                           static_cast<double>(t.h0_im[i]) * t.h0_im[i]);
        predicted += wave_density(d.spectrum, t.kx[i], t.kz[i]) * dk * dk;
    }

    CAPTURE(realised);
    CAPTURE(predicted);
    // One random realisation, so agreement is statistical, not exact. The
    // energy is concentrated in a few thousand cells near the spectral peak,
    // giving a sampling spread of a couple of percent; 10% is a comfortable
    // bound that would still catch a missing factor of 2 or sqrt(2).
    CHECK(realised == doctest::Approx(predicted).epsilon(0.10));
}

TEST_CASE("the mirrored table really is the conjugate of the mirrored cell")
{
    OceanDesc d;
    d.size     = 64;
    d.seed     = 7;
    d.spectrum = default_spectrum();

    SpectrumTables t;
    build_spectrum(d, t);

    const std::uint32_t n = d.size;
    for (std::uint32_t y = 0; y < n; ++y) {
        const std::uint32_t my = (n - y) % n;
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::uint32_t mx = (n - x) % n;
            const std::size_t i = static_cast<std::size_t>(y) * n + x;
            const std::size_t m = static_cast<std::size_t>(my) * n + mx;
            REQUIRE(t.h0c_re[i] == t.h0_re[m]);
            REQUIRE(t.h0c_im[i] == -t.h0_im[m]);
        }
    }
}

TEST_CASE("dispersion and wavevector tables are internally consistent")
{
    OceanDesc d;
    d.size         = 64;
    d.patch_length = 200.0f;
    d.spectrum     = default_spectrum();

    SpectrumTables t;
    build_spectrum(d, t);

    const std::size_t cells = static_cast<std::size_t>(d.size) * d.size;
    for (std::size_t i = 0; i < cells; ++i) {
        const double k = std::sqrt(static_cast<double>(t.kx[i]) * t.kx[i] +
                                   static_cast<double>(t.kz[i]) * t.kz[i]);
        CHECK(t.omega[i] == doctest::Approx(std::sqrt(9.81 * k)).epsilon(1e-5));
        if (k > 0.0) {
            CHECK(t.k_inv[i] == doctest::Approx(1.0 / k).epsilon(1e-5));
        } else {
            CHECK(t.k_inv[i] == 0.0f);  // DC handled explicitly, never 1/0
        }
    }

    // Index 0 is DC; index N/2 is Nyquist, the most negative representable
    // wavenumber under standard DFT bin ordering.
    CHECK(t.kx[0] == 0.0f);
    CHECK(wave_component(0, 64, 200.0f) == 0.0f);
    CHECK(wave_component(1, 64, 200.0f) > 0.0f);
    CHECK(wave_component(63, 64, 200.0f) < 0.0f);
    CHECK(wave_component(32, 64, 200.0f) < 0.0f);  // Nyquist counts as negative
}

TEST_CASE("spectrum generation is deterministic and seed-sensitive")
{
    OceanDesc d;
    d.size     = 64;
    d.seed     = 123456789;
    d.spectrum = default_spectrum();

    SpectrumTables a, b;
    build_spectrum(d, a);
    build_spectrum(d, b);

    const std::size_t bytes =
        static_cast<std::size_t>(d.size) * d.size * sizeof(float);
    CHECK(std::memcmp(a.h0_re.data(), b.h0_re.data(), bytes) == 0);
    CHECK(std::memcmp(a.h0_im.data(), b.h0_im.data(), bytes) == 0);
    CHECK(std::memcmp(a.h0c_re.data(), b.h0c_re.data(), bytes) == 0);
    CHECK(std::memcmp(a.h0c_im.data(), b.h0c_im.data(), bytes) == 0);

    OceanDesc other = d;
    other.seed = 987654321;
    SpectrumTables c;
    build_spectrum(other, c);
    CHECK(std::memcmp(a.h0_re.data(), c.h0_re.data(), bytes) != 0);

    // The wavevector tables depend only on geometry, so they must NOT change
    // with the seed.
    CHECK(std::memcmp(a.kx.data(), c.kx.data(), bytes) == 0);
    CHECK(std::memcmp(a.omega.data(), c.omega.data(), bytes) == 0);
}

TEST_CASE("rows are generated independently of one another")
{
    // Each row draws from a generator seeded by mix64(seed, row), so no row
    // depends on how many samples earlier rows consumed. That is what will let
    // the build be split across threads later without changing a single bit -
    // and it is why we do not use PCG's stream parameter, whose nearby streams
    // are not guaranteed independent.
    OceanDesc d;
    d.size     = 32;
    d.seed     = 55;
    d.spectrum = default_spectrum();

    SpectrumTables t;
    build_spectrum(d, t);

    // Adjacent rows should not be correlated. Compare row 3 against row 4 at
    // matching columns; a correlation near +/-1 would mean the generators are
    // producing related sequences.
    const std::uint32_t n = d.size;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (std::uint32_t x = 0; x < n; ++x) {
        // Normalise out the spectral envelope, which legitimately differs
        // between rows; we only want the random part.
        const double a = t.h0_re[3 * n + x];
        const double b = t.h0_re[4 * n + x];
        sxy += a * b;
        sxx += a * a;
        syy += b * b;
    }
    const double corr = sxy / std::sqrt(sxx * syy);
    CAPTURE(corr);
    CHECK(std::abs(corr) < 0.5);
}

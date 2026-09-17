#include "core/spectrum.hpp"

#include "core/rng.hpp"

#include <cmath>

namespace ocean::detail {
namespace {

constexpr double kPi    = 3.14159265358979323846;
constexpr double kTwoPi = 6.28318530717958647692;

// Wrap an angle into [-pi, pi).
double wrap_angle(double a) noexcept
{
    a = std::fmod(a + kPi, kTwoPi);
    if (a < 0.0) a += kTwoPi;
    return a - kPi;
}

}  // namespace

// ---------------------------------------------------------------------------
// JONSWAP
// ---------------------------------------------------------------------------
//
// JONSWAP (Hasselmann et al., Joint North Sea Wave Project, 1973) is
// Pierson-Moskowitz with a peak-sharpening factor. PM describes a *fully
// developed* sea - wind has blown long enough over enough water that the waves
// have stopped growing. Real coastal water is usually fetch-limited instead,
// and a fetch-limited sea has a sharper, taller spectral peak than PM predicts.
// JONSWAP adds the factor gamma^r to model exactly that.
//
// This is why we take fetch as a parameter at all: it is what separates a
// young, steep, short-crested wind sea from a mature swell, and it is the knob
// that Phillips (Tessendorf's original choice) simply does not have.

double jonswap_peak_omega(const SpectrumDesc& d)
{
    // omega_p = 22 * (g^2 / (U * F))^(1/3).
    //
    // Equivalently 22 * (g/U) * xtilde^(-1/3) with the dimensionless fetch
    // xtilde = g*F/U^2. Longer fetch or stronger wind pushes the peak to lower
    // frequency, i.e. longer waves - which matches the intuition that open
    // ocean swell is long and lake chop is short.
    const double g = d.gravity;
    const double u = static_cast<double>(d.wind_speed);
    const double f = static_cast<double>(d.fetch);
    if (u <= 0.0 || f <= 0.0) return 0.0;
    return 22.0 * std::cbrt(g * g / (u * f));
}

double jonswap_alpha(const SpectrumDesc& d)
{
    // alpha = 0.076 * xtilde^(-0.22), xtilde = g*F/U^2.
    const double g = d.gravity;
    const double u = static_cast<double>(d.wind_speed);
    const double f = static_cast<double>(d.fetch);
    if (u <= 0.0 || f <= 0.0) return 0.0;
    const double x_tilde = g * f / (u * u);
    return 0.076 * std::pow(x_tilde, -0.22);
}

double jonswap(const SpectrumDesc& d, double omega)
{
    if (!(omega > 0.0)) return 0.0;

    const double g       = d.gravity;
    const double omega_p = jonswap_peak_omega(d);
    if (!(omega_p > 0.0)) return 0.0;

    const double alpha = jonswap_alpha(d);
    const double gamma = static_cast<double>(d.peak_enhancement);

    // Pierson-Moskowitz base: (alpha g^2 / omega^5) * exp(-5/4 (omega_p/omega)^4).
    //
    // The omega^-5 tail is the Phillips equilibrium range - the shape a
    // saturated, breaking-limited sea settles into. The exponential cuts off
    // everything below the peak: waves longer than the peak cannot exist
    // because the wind has not blown long enough to build them.
    //
    // Note the exponential dominates as omega -> 0, so despite the omega^-5
    // the density goes to zero there rather than diverging. There is no
    // singularity to guard except omega == 0 exactly.
    const double wp_over_w = omega_p / omega;
    const double w2  = omega * omega;
    const double w4  = w2 * w2;
    const double pm  = (alpha * g * g / (w4 * omega)) *
                       std::exp(-1.25 * std::pow(wp_over_w, 4.0));

    // Peak enhancement gamma^r, with r a Gaussian bump centred on omega_p.
    // sigma is asymmetric (0.07 below the peak, 0.09 above) because the
    // measured JONSWAP peak is skewed - it rises more steeply than it falls.
    const double sigma = (omega <= omega_p) ? 0.07 : 0.09;
    const double dw    = omega - omega_p;
    const double r     = std::exp(-(dw * dw) /
                                  (2.0 * sigma * sigma * omega_p * omega_p));

    return pm * std::pow(gamma, r);
}

// ---------------------------------------------------------------------------
// Directional spreading
// ---------------------------------------------------------------------------

double directional_spread(const SpectrumDesc& d, double omega, double theta)
{
    const double omega_p = jonswap_peak_omega(d);
    if (!(omega > 0.0) || !(omega_p > 0.0)) return 0.0;

    const double g = d.gravity;
    const double u = static_cast<double>(d.wind_speed);
    const double ratio = omega / omega_p;

    // Hasselmann's frequency-dependent spreading exponent. Energy at the peak
    // is the most tightly beamed around the wind direction; both longer and
    // shorter waves spread more widely. That frequency dependence is the whole
    // reason to use a spreading *function* rather than a single fixed cone -
    // it is what produces short-crested, patchy-looking water instead of
    // corduroy.
    double s;
    if (ratio < 1.0) {
        s = 6.97 * std::pow(ratio, 4.06);
    } else {
        const double mu = -2.33 - 1.45 * ((u * omega_p / g) - 1.17);
        s = 9.77 * std::pow(ratio, mu);
    }

    // Swell sharpening.
    //
    // HONEST CAVEAT: this term is a plausible reconstruction of Horvath 2015's
    // swell parameter, not a verified transcription. The shape (a tanh-weighted
    // boost to the spreading exponent, strongest well below the peak, scaling
    // with the square of the control) is right in character, but the constant
    // 16 should be checked against the paper before anyone relies on it as
    // "the Horvath model". It behaves correctly as an artist control either
    // way: 0 leaves the wind-sea spreading untouched, 1 collapses the spectrum
    // toward a narrow, nearly unidirectional swell.
    const double swell = static_cast<double>(d.swell);
    if (swell > 0.0) {
        s += 16.0 * std::tanh(omega_p / omega) * swell * swell;
    }

    if (s < 0.0) s = 0.0;

    // Positive-cosine-squared form: D(theta) = Q(s) * cos^(2s)((theta-theta_w)/2).
    //
    // The HALF angle matters. cos^(2s)(theta - theta_w) would be symmetric
    // front-to-back and generate waves travelling into the wind with the same
    // energy as waves travelling with it. The half-angle form vanishes exactly
    // at theta = theta_w +/- pi, so upwind waves get zero energy for free -
    // no extra suppression term needed.
    const double dtheta = wrap_angle(theta - static_cast<double>(d.wind_direction));
    const double c = std::cos(0.5 * dtheta);  // >= 0 after wrapping
    if (c <= 0.0) return 0.0;

    // Normalisation Q(s) = Gamma(s+1) / (2*sqrt(pi)*Gamma(s+1/2)), from
    //   integral over [-pi,pi] of cos^(2s)(theta/2) dtheta
    //     = 2*sqrt(pi) * Gamma(s+1/2) / Gamma(s+1).
    //
    // Q(s) cannot be skipped even though it only scales the result: s varies
    // with omega, so without it the *relative* energy between frequencies would
    // be wrong, not merely the overall level. Computed via lgamma to avoid
    // overflow - Gamma(s+1) itself overflows a double for modest s.
    const double q = std::exp(std::lgamma(s + 1.0) - std::lgamma(s + 0.5)) /
                     (2.0 * std::sqrt(kPi));

    return q * std::pow(c, 2.0 * s);
}

// ---------------------------------------------------------------------------
// Frequency space -> wavenumber space
// ---------------------------------------------------------------------------

double wave_density(const SpectrumDesc& d, double kx, double kz)
{
    const double k2 = kx * kx + kz * kz;
    if (!(k2 > 0.0)) return 0.0;  // DC bin carries no wave energy

    const double k = std::sqrt(k2);
    const double g = d.gravity;

    // Deep-water dispersion. Valid while depth > ~L/2, which holds for open
    // ocean at every wavelength we represent. Shallow water would need the
    // full omega^2 = g*k*tanh(k*h); that is a V2 concern.
    const double omega = std::sqrt(g * k);

    const double s_omega = jonswap(d, omega);
    if (!(s_omega > 0.0)) return 0.0;

    // Two changes of variable, and both are easy to get wrong:
    //
    // 1. Frequency -> wavenumber. S(k) dk = S(omega) domega, so we multiply by
    //    the Jacobian domega/dk. With omega = sqrt(g*k), domega/dk = g/(2*omega).
    //
    // 2. Polar -> Cartesian. The 2D density must satisfy
    //       integral Psi(kx,kz) dkx dkz = integral S(k) D(theta) dk dtheta,
    //    and dkx dkz = k dk dtheta, so Psi = S(k) * D(theta) / k.
    //
    // Forgetting the 1/k is the classic bug here: the surface still looks like
    // an ocean, but the energy balance across scales is wrong, so it never
    // quite matches a real sea state at any wind speed.
    const double domega_dk = g / (2.0 * omega);
    const double theta     = std::atan2(kz, kx);
    const double spread    = directional_spread(d, omega, theta);

    double psi = s_omega * domega_dk * spread / k;

    // Tessendorf's short-wave suppression, exp(-k^2 l^2). Waves below a couple
    // of grid cells cannot be represented and only alias into shimmer, so we
    // roll them off smoothly rather than letting them fight the sampling rate.
    // At k = 2*pi/cutoff the factor is exp(-1), so `small_wave_cutoff` is a
    // soft knee, not a hard wall.
    const double l = static_cast<double>(d.small_wave_cutoff) / kTwoPi;
    psi *= std::exp(-k2 * l * l);

    return psi;
}

// ---------------------------------------------------------------------------
// Table construction
// ---------------------------------------------------------------------------

void SpectrumTables::allocate(std::uint32_t size)
{
    n = size;
    const std::size_t cells = static_cast<std::size_t>(size) * size;
    h0_re  = AlignedBuffer<float>(cells);
    h0_im  = AlignedBuffer<float>(cells);
    h0c_re = AlignedBuffer<float>(cells);
    h0c_im = AlignedBuffer<float>(cells);
    omega  = AlignedBuffer<float>(cells);
    kx     = AlignedBuffer<float>(cells);
    kz     = AlignedBuffer<float>(cells);
    k_inv  = AlignedBuffer<float>(cells);
}

void build_spectrum(const OceanDesc& desc, SpectrumTables& out)
{
    const std::uint32_t n = desc.size;
    const float L = desc.patch_length;
    out.allocate(n);

    // Grid cell area in k-space. The discrete amplitude in one bin must carry
    // the energy the continuous density assigns to that bin's area, which is
    // where dk^2 comes from below.
    const double dk = kTwoPi / static_cast<double>(L);

    for (std::uint32_t y = 0; y < n; ++y) {
        // A generator that is a pure function of (seed, row): rows may be
        // built in any order, on any thread, for bit-identical output.
        Pcg32 rng = row_rng(desc.seed, y);
        const float kz_row = wave_component(y, n, L);

        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * n + x;
            const float kx_val = wave_component(x, n, L);

            out.kx[i] = kx_val;
            out.kz[i] = kz_row;

            const double k2 = static_cast<double>(kx_val) * kx_val +
                              static_cast<double>(kz_row) * kz_row;
            const double k  = std::sqrt(k2);

            out.omega[i] = static_cast<float>(std::sqrt(desc.spectrum.gravity * k));
            out.k_inv[i] = (k > 0.0) ? static_cast<float>(1.0 / k) : 0.0f;

            const double psi = wave_density(desc.spectrum, kx_val, kz_row);

            // h0(k) = (1/sqrt(2)) * (xi_r + i*xi_i) * sqrt(P(k)),
            // with P(k) = Psi(k) * dkx * dkz / 2.
            //
            // Where the /2 comes from: the time-dependent amplitude is
            //     h(k,t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t},
            // and since h0(k) and h0(-k) are independent zero-mean draws, the
            // cross terms vanish in expectation and
            //     E|h(k,t)|^2 = P(k) + P(-k) = 2 P(k).
            // Summing that over the grid must equal the surface variance, so
            // each bin gets half the density it would otherwise carry. Drop the
            // /2 and the whole ocean comes out sqrt(2) times too tall - a bug
            // that looks plausible and survives visual inspection indefinitely.
            //
            // Collecting the constants: amp = 0.5 * sqrt(Psi) * dk.
            const double amp = 0.5 * std::sqrt(psi) * dk;

            // Exactly one Box-Muller pair per cell: the real and imaginary
            // parts are the two independent Gaussians the formulation calls
            // for, so nothing is cached and nothing is wasted.
            const GaussianPair g = next_gaussian_pair(rng);
            out.h0_re[i] = static_cast<float>(g.a * amp);
            out.h0_im[i] = static_cast<float>(g.b * amp);
        }
    }

    // Second pass: conj(h0(-k)).
    //
    // Done once here, at init, so the per-frame loop reads cell i and cell i
    // only. Computing the mirror index every frame would turn the hottest loop
    // in the library into a scattered gather across the entire grid.
    for (std::uint32_t y = 0; y < n; ++y) {
        const std::uint32_t my = (n - y) % n;  // index of -kz
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::uint32_t mx = (n - x) % n;  // index of -kx
            const std::size_t i = static_cast<std::size_t>(y) * n + x;
            const std::size_t m = static_cast<std::size_t>(my) * n + mx;
            out.h0c_re[i] =  out.h0_re[m];
            out.h0c_im[i] = -out.h0_im[m];
        }
    }
}

}  // namespace ocean::detail

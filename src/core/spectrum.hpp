// oceanlib - internal header. Not part of the public API.
//
// Directional wave spectrum: JONSWAP energy density with directional
// spreading, mapped from frequency space into the 2D wavenumber grid, and the
// time-independent initial amplitudes h0(k) that every frame is built from.
//
// References:
//   Tessendorf 2001, "Simulating Ocean Water"  - the FFT surface formulation
//   Hasselmann et al. 1973                     - the JONSWAP spectrum
//   Hasselmann et al. 1980 / Mitsuyasu 1975    - directional spreading
//   Horvath 2015, "Empirical Directional Wave Spectra for Computer Graphics"
#pragma once

#include "core/aligned.hpp"
#include "ocean/ocean.hpp"

#include <cstdint>

namespace ocean::detail {

// ---------------------------------------------------------------------------
// Analytic spectrum, exposed so tests can integrate it independently of the
// discretised tables below.
//
// Everything here takes and returns double. It runs once at init, never per
// frame, so precision is free - and the k-space conversion below divides by
// small quantities, where float would lose real accuracy.
// ---------------------------------------------------------------------------

// Finite-depth dispersion relation, omega^2 = g k tanh(k h).
//
// `depth <= 0` selects the deep-water limit exactly (tanh(kh) -> 1 as
// h -> infinity), which is the default and matches Tessendorf's original
// omega = sqrt(g k) bit for bit - this is not an approximation of the deep
// branch, it IS the deep branch, taken as a literal special case rather than
// evaluated as a tanh() that happens to be very close to 1. That keeps every
// existing deep-water test and every existing deep-water ocean unaffected by
// this feature's existence.
//
// Exposed here (not just used internally) so tests can check the shallow and
// deep limits, and the derivative below, against closed-form textbook results
// rather than only against each other.
double dispersion_omega(double gravity, double k, double depth) noexcept;

// d(omega)/dk for the same relation - the group velocity, and the Jacobian
// needed for the frequency-to-wavenumber change of variables in
// wave_density(). Deep water: g/(2*omega). Finite depth, differentiating
// omega^2 = g k tanh(kh) implicitly:
//
//   2 omega (domega/dk) = g [tanh(kh) + kh sech^2(kh)]
//   domega/dk = g [tanh(kh) + kh sech^2(kh)] / (2 omega)
//
// which reduces to exactly g/(2*omega) as depth -> infinity (tanh -> 1,
// sech^2 -> 0), matching dispersion_omega's own exact deep-water branch.
// sech^2(kh) is computed as 1 - tanh(kh)^2 rather than 1/cosh(kh)^2, because
// cosh(kh) overflows a double for kh above a few hundred while tanh saturates
// safely to 1.0 - a real concern here, since a coastal depth combined with a
// short-wave wavenumber routinely makes kh reach into the thousands.
double dispersion_domega_dk(double gravity, double k, double omega,
                            double depth) noexcept;

// Peak angular frequency [rad/s]. omega_p = 22 * (g^2 / (U*F))^(1/3).
double jonswap_peak_omega(const SpectrumDesc& d);

// Equilibrium-range constant (dimensionless). alpha = 0.076 * (U^2/(g*F))^0.22.
double jonswap_alpha(const SpectrumDesc& d);

// Non-directional JONSWAP energy density S(omega), in m^2 / (rad/s).
double jonswap(const SpectrumDesc& d, double omega);

// Normalised directional spreading D(omega, theta), in 1/rad.
// Integrates to 1 over theta in [-pi, pi) for every omega.
double directional_spread(const SpectrumDesc& d, double omega, double theta);

// Full 2D wavenumber spectral density Psi(kx, kz), in m^2 / (rad/m)^2.
// Integrating it over the kx-kz plane gives the variance of the surface.
double wave_density(const SpectrumDesc& d, double kx, double kz);

// Signed wavenumber component for grid index i.
//
// Indices [0, N/2) map to positive wavenumbers and [N/2, N) to negative ones -
// standard DFT bin ordering, so the FFT output needs no fftshift. Using the
// "centred" convention (i - N/2) instead would offset the result by half a
// period, which shows up as a checkerboard sign flip across the grid.
inline float wave_component(std::uint32_t i, std::uint32_t n,
                            float patch_length) noexcept
{
    constexpr float kTwoPi = 6.28318530717958647692f;
    const int signed_i = (i < n / 2) ? static_cast<int>(i)
                                     : static_cast<int>(i) - static_cast<int>(n);
    return kTwoPi * static_cast<float>(signed_i) / patch_length;
}

// ---------------------------------------------------------------------------
// Discretised, precomputed per-cell tables
// ---------------------------------------------------------------------------

// Everything about the surface that does not depend on time. Built once in the
// constructor; the per-frame path only reads it.
struct SpectrumTables {
    std::uint32_t n = 0;

    // h0(k): the initial complex amplitude of the wave travelling along +k.
    AlignedBuffer<float> h0_re, h0_im;

    // conj(h0(-k)): the amplitude of the wave travelling along -k, already
    // conjugated and mirrored so the per-frame loop is a straight indexed read
    // instead of a mirror-index computation (which would be a scattered access
    // across the whole grid, every frame, for every cell).
    AlignedBuffer<float> h0c_re, h0c_im;

    // omega(k) per cell, from dispersion_omega() - deep water unless
    // desc.spectrum.depth is set to a finite positive value.
    AlignedBuffer<float> omega;

    // Wavevector and 1/|k| (defined as 0 at the DC bin). Stored rather than
    // recomputed because the per-frame loop is memory-bound; three arrays is
    // the minimum from which all eight field spectra can be derived cheaply.
    AlignedBuffer<float> kx, kz, k_inv;

    void allocate(std::uint32_t size);
};

// Fills `out` for the given descriptor. Deterministic: a pure function of
// (desc.seed, desc), independent of the order rows are visited in.
void build_spectrum(const OceanDesc& desc, SpectrumTables& out);

}  // namespace ocean::detail

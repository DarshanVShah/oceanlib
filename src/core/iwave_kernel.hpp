// oceanlib - internal header. Not part of the public API.
//
// The iWave convolution kernel: a finite-support real-space stencil whose
// Fourier symbol approximates the linearised deep-water surface operator.
//
// WHY A CONVOLUTION AT ALL
//
// Linearised, irrotational, inviscid free-surface flow gives
//
//     d^2h/dt^2 = -g * G{h},      symbol of G = |k|
//
// (Laplace in the volume, kinematic + dynamic boundary conditions linearised
// onto y = 0; substituting a plane wave recovers omega^2 = g|k|, the same
// deep-water dispersion relation the FFT ocean in spectrum.hpp solves.)
//
// G is the operator sqrt(-laplacian). |k| is not a polynomial in k, so G is
// NOT a differential operator - it is nonlocal, and in real space it is a
// convolution with infinite support. iWave's whole idea is to truncate that
// convolution to a small radius P and apply it directly, which keeps the
// operator LOCAL. Locality is the entire product here: it is what allows a
// spatially varying absorbing layer and a spatially varying obstruction mask,
// neither of which exists in a global spectral multiply.
//
// Reference: Tessendorf, "Interactive Water Surfaces", Game Programming
// Gems 4 (2004).
#pragma once

#include "core/aligned.hpp"

#include <cstdint>

namespace ocean::detail {

// The symbol S(k) = omega^2(k) / g, in 1/m.
//
// Deep water (depth <= 0) is S(k) = k exactly, taken as a literal special
// case rather than evaluated as a tanh() that happens to be near 1 - the same
// convention, for the same reason, as dispersion_omega() in spectrum.hpp.
// Finite depth is S(k) = k * tanh(k * depth), so a shallow interaction field
// and a shallow FFT ocean stay consistent with each other for free.
//
// Writing the kernel derivation against a SYMBOL rather than against a
// hardcoded |k| is what makes that generalisation cost nothing.
double iwave_symbol(double k, double depth) noexcept;

// How the stencil is derived. Both are built at init and cost nothing per
// frame; the choice is measured, not assumed - see test_iwave_kernel.cpp,
// which reports the realised symbol error of each.
enum class KernelMethod {
    // Tessendorf's own derivation: the inverse Hankel transform of the symbol,
    // evaluated numerically out to the grid Nyquist and truncated at radius P.
    // This is the reference implementation, and the cross-check that the
    // least-squares fit below is not fitting nonsense.
    Hankel,

    // Solve directly for the truncated stencil whose realised symbol best
    // matches the true symbol over the resolved band, in a relative-error
    // weighted least-squares sense.
    //
    // The reasoning: truncating the Hankel transform compromises the symbol,
    // and the symbol is the only thing about this operator that matters - it
    // IS the dispersion relation. So rather than truncate and accept whatever
    // symbol error falls out, solve for the best symbol the same support can
    // represent. Same cost per frame, strictly better dispersion, and the
    // improvement is a measured number rather than a claim.
    LeastSquares,
};

// A derived stencil, plus everything about it the solver needs to know.
//
// The dense (2P+1)^2 form is what the hot loop reads: the inner loop wants
// straight indexed taps with no symmetry branching, so D4 symmetry is used to
// DERIVE and to STORE-distinctly, not to save work per tap.
struct IWaveKernel {
    std::uint32_t radius = 0;    // P
    std::uint32_t width  = 0;    // 2P+1
    float         cell   = 0.0f; // metres per cell, the dx the fit assumed

    // (2P+1)^2 taps, row-major over (j + P) * width + (i + P).
    AlignedBuffer<float> taps;

    // Largest value the REALISED symbol reaches anywhere in the resolved band.
    //
    // Not the ideal Nyquist value: truncation ringing can push the realised
    // symbol above the true one, and the leapfrog stability bound depends on
    // the largest frequency actually present, which is set by the kernel we
    // actually built. Computed here so the solver's dt limit is derived from
    // the real thing rather than from theory.
    double max_symbol = 0.0;

    // Low edge of the band this kernel claims, in rad/m. Below it the
    // even-stencil-versus-|k| mismatch dominates for ANY finite stencil and
    // waves run slow; above it the kernel is held to peak_relative_error.
    // Long waves below this edge are the FFT ocean's job.
    double band_low = 0.0;

    // max |realised/true - 1| over [band_low, grid Nyquist]. This is the
    // dispersion error the truncation costs, and it is the number that
    // justifies a particular kernel_radius.
    //
    // It is a SYMBOL error; the wave-speed error is half of it, because
    // omega = sqrt(g*S) and so d(omega)/omega = (1/2) dS/S.
    double peak_relative_error = 0.0;

    // Smallest value the realised symbol reaches anywhere the grid can
    // represent, corners of the square included.
    //
    // A meaningfully negative value is not a quality problem, it is a
    // correctness one: omega^2 = g*S, so S < 0 means that mode grows
    // exponentially instead of oscillating, and no choice of timestep can
    // stabilise it. See kernel_is_stable() below.
    double min_symbol = 0.0;

    [[nodiscard]] float tap(int i, int j) const noexcept
    {
        const int p = static_cast<int>(radius);
        return taps.data()[static_cast<std::size_t>(j + p) * width + (i + p)];
    }
};

// Build a stencil of radius P for a grid of `cell` metres, for the given
// water depth (<= 0 = deep).
//
// `fit_samples` is the resolution of the k-plane sample set the LeastSquares
// method fits over (per axis, over one D4 octant); ignored by Hankel. It has
// no per-frame cost and the default is far past the point of diminishing
// returns - the symbol and the basis are both smooth.
// `band_low_frac` (LeastSquares only) is the low edge of the band the fit is
// held to, as a fraction of the grid Nyquist. It is a real design parameter,
// not a tuning constant: a finite stencil cannot match |k| over an unbounded
// range of scales, so asking it to try at very low k costs accuracy at the
// short wavelengths this field exists to carry. See the comment at the fit
// itself for the measurement that established this.
void build_iwave_kernel(std::uint32_t radius, double cell, double depth,
                        KernelMethod method, IWaveKernel& out,
                        std::uint32_t fit_samples = 96,
                        double band_low_frac = 0.125);

// The symbol the stencil ACTUALLY realises: sum_ij G_ij cos(k . x_ij).
//
// Real by construction, because the stencil is even in both axes. Exposed so
// tests can measure dispersion error directly against iwave_symbol() rather
// than inferring it from a simulation.
double realised_symbol(const IWaveKernel& k, double kx, double kz) noexcept;

// Whether this kernel can be used at all.
//
// Measured, not assumed: truncating the inverse Hankel transform (Tessendorf's
// own derivation) produces a symbol that is faithful throughout the inscribed
// Nyquist disc but dips to roughly -0.12..-0.18 of its maximum in the CORNERS
// of k-space, at every radius tested from 2 to 12. Those corner modes are
// shorter than two cells along the diagonal - aliasing artefacts rather than
// waves - but the grid still holds them, and a negative symbol makes them grow
// exponentially. Nothing deliberately excites a checkerboard, which is exactly
// what makes it dangerous: rounding noise seeds it, damping cannot remove an
// exponential, and it surfaces later as "the water sometimes explodes".
//
// Fitting over a band removes it for free, because the fit includes the
// corners at low weight - enough to hold the symbol non-negative there without
// spending accuracy in the band that matters.
//
// The builder deliberately does NOT throw on a failing kernel: it is a
// derivation and analysis tool, and the tests need to construct the bad kernel
// in order to measure it. Rejection belongs where user-facing configuration is
// validated, in the solver's constructor.
[[nodiscard]] inline bool kernel_is_stable(const IWaveKernel& k) noexcept
{
    return k.min_symbol >= -1e-3 * k.max_symbol && k.max_symbol > 0.0;
}

// Bessel J0, needed by the Hankel derivation. Abramowitz & Stegun 9.4.1 and
// 9.4.3; accurate to about 5e-8, which is far inside float. In here rather
// than <cmath> because std::cyl_bessel_j is a C++17 special-math function
// MSVC does not implement, and a dependency is not an option (ADR-001).
double bessel_j0(double x) noexcept;

}  // namespace ocean::detail

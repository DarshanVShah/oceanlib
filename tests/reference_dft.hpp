#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

// Naive reference transforms, for validating the FFT. Test-only: these never
// ship in the library.
//
// Computed in double while the FFT runs in float, so the reference is
// trustworthy to ~1e-16 and any disagreement we see is the FFT's error, not a
// contest between two equally shaky float computations.
namespace refdft {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// sign = -1 forward, +1 inverse. No 1/N normalisation, matching FftPlan.
inline void dft_1d(const float* re, const float* im, std::uint32_t n, int sign,
                   std::vector<double>& out_re, std::vector<double>& out_im)
{
    out_re.assign(n, 0.0);
    out_im.assign(n, 0.0);
    for (std::uint32_t k = 0; k < n; ++k) {
        double sr = 0.0, si = 0.0;
        for (std::uint32_t j = 0; j < n; ++j) {
            // Reduce the phase index mod n before scaling: keeps the angle in
            // [0, 2pi) so we never lean on library argument reduction.
            const std::uint64_t idx =
                (static_cast<std::uint64_t>(j) * k) % n;
            const double a = sign * kTwoPi * static_cast<double>(idx) /
                             static_cast<double>(n);
            const double c = std::cos(a);
            const double s = std::sin(a);
            const double xr = re[j];
            const double xi = im[j];
            sr += xr * c - xi * s;
            si += xr * s + xi * c;
        }
        out_re[k] = sr;
        out_im[k] = si;
    }
}

// Full O(N^4) 2D transform: the direct quadruple sum, NOT row-column.
//
// That is the whole point of this reference. FftPlan::transform_2d assumes the
// kernel is separable and does rows-then-columns; if we validated it against a
// reference that made the same assumption, a bug in that assumption would
// cancel out on both sides and the test would pass. Summing over all four
// indices independently shares no structure with the thing under test.
inline void dft_2d(const float* re, const float* im, std::uint32_t n, int sign,
                   std::vector<double>& out_re, std::vector<double>& out_im)
{
    const std::size_t cells = static_cast<std::size_t>(n) * n;
    out_re.assign(cells, 0.0);
    out_im.assign(cells, 0.0);

    for (std::uint32_t ky = 0; ky < n; ++ky) {
        for (std::uint32_t kx = 0; kx < n; ++kx) {
            double sr = 0.0, si = 0.0;
            for (std::uint32_t y = 0; y < n; ++y) {
                for (std::uint32_t x = 0; x < n; ++x) {
                    const std::uint64_t idx =
                        (static_cast<std::uint64_t>(kx) * x +
                         static_cast<std::uint64_t>(ky) * y) % n;
                    const double a = sign * kTwoPi * static_cast<double>(idx) /
                                     static_cast<double>(n);
                    const double c = std::cos(a);
                    const double s = std::sin(a);
                    const std::size_t i = static_cast<std::size_t>(y) * n + x;
                    const double xr = re[i];
                    const double xi = im[i];
                    sr += xr * c - xi * s;
                    si += xr * s + xi * c;
                }
            }
            const std::size_t o = static_cast<std::size_t>(ky) * n + kx;
            out_re[o] = sr;
            out_im[o] = si;
        }
    }
}

// Worst-case absolute difference, divided by the largest reference magnitude.
// Relative-to-peak rather than per-element relative: a DFT bin that is
// legitimately near zero would make per-element relative error meaningless.
inline double relative_error(const float* re, const float* im,
                             const std::vector<double>& ref_re,
                             const std::vector<double>& ref_im)
{
    double worst = 0.0, scale = 0.0;
    for (std::size_t i = 0; i < ref_re.size(); ++i) {
        const double dr = static_cast<double>(re[i]) - ref_re[i];
        const double di = static_cast<double>(im[i]) - ref_im[i];
        worst = std::max(worst, std::sqrt(dr * dr + di * di));
        scale = std::max(scale, std::sqrt(ref_re[i] * ref_re[i] +
                                          ref_im[i] * ref_im[i]));
    }
    return scale > 0.0 ? worst / scale : worst;
}

}  // namespace refdft

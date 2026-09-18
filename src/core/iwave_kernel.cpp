#include "core/iwave_kernel.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ocean::detail {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Number of Simpson intervals for the Hankel integral.
//
// The integrand S(k) J0(k r) k oscillates with period 2*pi/r in k. The widest
// radius is P*dx and the cutoff is pi/dx, so the product k_max * r_max is
// exactly pi*P regardless of dx or extent - about 19 radians, i.e. three
// oscillations, for P = 6. A few thousand intervals is far past sufficient,
// and this runs once at construction.
constexpr int kHankelIntervals = 8192;

// Orbit size of the D4 group acting on the offset (a, b), a >= b >= 0.
//
// (0,0) is fixed. An on-axis offset (a,0) has the four images (+-a,0),(0,+-a).
// A diagonal offset (a,a) has the four images (+-a,+-a). Anything else has the
// full eight. Getting this wrong would silently misweight the fit, so it is
// derived once here and reused by both the fit and the expansion.
int orbit_size(int a, int b) noexcept
{
    if (a == 0) return 1;              // implies b == 0
    if (b == 0) return 4;              // on-axis
    if (a == b) return 4;              // on-diagonal
    return 8;
}

// The D4 orbit of (a, b), written into `out`. Returns the count.
int orbit_of(int a, int b, int (*out)[2]) noexcept
{
    int n = 0;
    if (a == 0) {
        out[n][0] = 0; out[n][1] = 0; ++n;
        return n;
    }
    if (b == 0) {
        out[n][0] =  a; out[n][1] =  0; ++n;
        out[n][0] = -a; out[n][1] =  0; ++n;
        out[n][0] =  0; out[n][1] =  a; ++n;
        out[n][0] =  0; out[n][1] = -a; ++n;
        return n;
    }
    const int sx[2] = {a, -a};
    const int sz[2] = {b, -b};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            out[n][0] = sx[i]; out[n][1] = sz[j]; ++n;
        }
    }
    if (a != b) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                out[n][0] = sz[j]; out[n][1] = sx[i]; ++n;
            }
        }
    }
    return n;
}

// Cholesky solve of a small symmetric positive-definite system, in place.
// Returns false if the matrix is not positive definite, which the caller
// treats as a hard error rather than papering over: a non-PD normal matrix
// would mean the basis is degenerate and the resulting "fit" meaningless.
bool cholesky_solve(std::vector<double>& a, std::vector<double>& b, int n)
{
    const std::size_t sn = static_cast<std::size_t>(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j)];
            for (int k = 0; k < j; ++k) {
                s -= a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(k)] *
                     a[static_cast<std::size_t>(j) * sn + static_cast<std::size_t>(k)];
            }
            if (i == j) {
                if (!(s > 0.0)) return false;
                a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)] = std::sqrt(s);
            } else {
                a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j)] =
                    s / a[static_cast<std::size_t>(j) * sn + static_cast<std::size_t>(j)];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int k = 0; k < i; ++k) {
            s -= a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(k)] *
                 b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] =
            s / a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = b[static_cast<std::size_t>(i)];
        for (int k = i + 1; k < n; ++k) {
            s -= a[static_cast<std::size_t>(k) * sn + static_cast<std::size_t>(i)] *
                 b[static_cast<std::size_t>(k)];
        }
        b[static_cast<std::size_t>(i)] =
            s / a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)];
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------

double iwave_symbol(double k, double depth) noexcept
{
    if (k <= 0.0) return 0.0;
    // Deep water is the literal special case, not a tanh() that happens to be
    // close to 1 - matching dispersion_omega()'s convention exactly, so a deep
    // interaction field is provably unaffected by finite-depth support
    // existing at all.
    if (depth <= 0.0) return k;
    return k * std::tanh(k * depth);
}

double bessel_j0(double x) noexcept
{
    // Abramowitz & Stegun 9.4.1 / 9.4.3. J0 is even, so fold first.
    x = std::fabs(x);
    if (x < 3.0) {
        const double y = (x / 3.0) * (x / 3.0);
        return 1.0 + y * (-2.2499997 +
               y * ( 1.2656208 +
               y * (-0.3163866 +
               y * ( 0.0444479 +
               y * (-0.0039444 +
               y * ( 0.0002100))))));
    }
    const double y = 3.0 / x;
    const double f0 = 0.79788456 +
        y * (-0.00000077 +
        y * (-0.00552740 +
        y * (-0.00009512 +
        y * ( 0.00137237 +
        y * (-0.00072805 +
        y * ( 0.00014476))))));
    const double t0 = x - 0.78539816 +
        y * (-0.04166397 +
        y * (-0.00003954 +
        y * ( 0.00262573 +
        y * (-0.00054125 +
        y * (-0.00029333 +
        y * ( 0.00013558))))));
    return f0 * std::cos(t0) / std::sqrt(x);
}

double realised_symbol(const IWaveKernel& k, double kx, double kz) noexcept
{
    const int    p  = static_cast<int>(k.radius);
    const double dx = k.cell;
    double sum = 0.0;
    for (int j = -p; j <= p; ++j) {
        for (int i = -p; i <= p; ++i) {
            // The stencil is even in both axes, so the imaginary halves of
            // e^{i k . x} cancel exactly and the symbol is real - which is
            // also why the operator is self-adjoint and cannot inject energy.
            sum += static_cast<double>(k.tap(i, j)) *
                   std::cos(kx * i * dx + kz * j * dx);
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------

void build_iwave_kernel(std::uint32_t radius, double cell, double depth,
                        KernelMethod method, IWaveKernel& out,
                        std::uint32_t fit_samples, double band_low_frac)
{
    if (radius < 1 || radius > 12) {
        throw std::invalid_argument(
            "ocean: kernel_radius must be in [1, 12], got " +
            std::to_string(radius));
    }
    if (!(cell > 0.0)) {
        throw std::invalid_argument(
            "ocean: interaction cell size must be positive");
    }

    const int    p     = static_cast<int>(radius);
    const int    width = 2 * p + 1;
    const double k_max = kPi / cell;   // grid Nyquist

    out.radius = radius;
    out.width  = static_cast<std::uint32_t>(width);
    out.cell   = static_cast<float>(cell);
    out.taps   = AlignedBuffer<float>(static_cast<std::size_t>(width) * width);

    // The distinct values under D4: (a, b) with p >= a >= b >= 0.
    std::vector<int> ai, bi;
    for (int a = 0; a <= p; ++a) {
        for (int b = 0; b <= a; ++b) { ai.push_back(a); bi.push_back(b); }
    }
    const int m = static_cast<int>(ai.size());
    std::vector<double> g(static_cast<std::size_t>(m), 0.0);

    if (method == KernelMethod::Hankel) {
        // Tessendorf's derivation. Because the symbol is radially symmetric,
        // its 2-D inverse transform collapses to a Hankel transform:
        //
        //     G(r) = (1/2pi) * integral_0^kmax  S(k) * J0(k r) * k dk
        //
        // The integral diverges for S(k) = k over an infinite range, which is
        // exactly the statement that the true kernel is singular at the
        // origin. Cutting it off at the grid Nyquist is not an approximation
        // of convenience - the grid genuinely cannot represent anything above
        // it.
        const double h = k_max / kHankelIntervals;
        for (int t = 0; t < m; ++t) {
            const double a = ai[static_cast<std::size_t>(t)];
            const double b = bi[static_cast<std::size_t>(t)];
            const double r = cell * std::sqrt(a * a + b * b);
            double acc = 0.0;
            for (int s = 0; s <= kHankelIntervals; ++s) {
                const double k = s * h;
                const double f = iwave_symbol(k, depth) * bessel_j0(k * r) * k;
                const double w = (s == 0 || s == kHankelIntervals)
                                     ? 1.0
                                     : ((s & 1) ? 4.0 : 2.0);
                acc += w * f;
            }
            // The cell-area factor is not cosmetic. The Hankel transform
            // returns the CONTINUUM kernel, a density in 1/m^2; the discrete
            // stencil approximates the convolution integral by a sum over
            // cells, so each tap carries the area of its cell:
            //
            //     sum_ij G_ij h_ij  ~=  integral G(x) h(x) d^2x  ~=  dx^2 * sum
            //
            // Omitting it leaves the realised symbol wrong by exactly 1/dx^2 -
            // which is a constant factor, so it survives every increase of P
            // and cannot be mistaken for truncation error.
            g[static_cast<std::size_t>(t)] =
                cell * cell * (acc * h / 3.0) / (2.0 * kPi);
        }
    } else {
        // Solve for the stencil whose realised symbol best matches the true
        // one over the resolved band.
        //
        // Basis: one function per distinct D4 value, summing cos(k . x) over
        // that value's whole orbit.
        //
        //     phi_ab(k) = sum_{(i,j) in orbit(a,b)} cos(kx*i*dx + kz*j*dx)
        //
        // The zero-sum constraint (the true symbol is exactly 0 at k = 0, so a
        // uniform water level must feel no restoring force) is enforced by
        // ELIMINATION, not by a penalty: setting
        //
        //     g_00 = -sum_{ab != 00} n_ab * g_ab
        //
        // turns the basis into psi_ab = phi_ab - n_ab, every member of which
        // vanishes at k = 0. The constraint then holds exactly, to the last
        // bit, rather than approximately.
        const int mm = m - 1;   // g_00 eliminated
        const std::size_t smm = static_cast<std::size_t>(mm);
        std::vector<double> ata(smm * smm, 0.0);
        std::vector<double> atb(smm, 0.0);
        std::vector<double> psi(smm, 0.0);

        // Weighting: relative error, restricted to the band this operator
        // is actually responsible for.
        //
        // The first attempt weighted 1/S^2 over the whole sampled square with
        // a floor near DC, and measured WORSE than Tessendorf's kernel in the
        // band that matters. The reason is instructive. A finite even stencil
        // has an analytic even symbol, so near k = 0 it behaves like c*k^2
        // while the true symbol behaves like k; that mismatch is unfixable at
        // any P. Weighting by 1/S^2 puts the HIGHEST weight exactly there, so
        // the fit spent its freedom on the region it cannot repair and paid
        // for it in the region it can. The floor limited the damage but did
        // not change the sign of the mistake.
        //
        // So the fit is now confined to a band:
        //
        //   |k| in [k_lo, k_max]   full relative weight - the usable band
        //   outside                a small weight, present only to keep the
        //                          symbol bounded and non-negative (a negative
        //                          realised symbol means omega^2 < 0, an
        //                          exponentially growing mode no timestep can
        //                          rescue), not to pull the fit
        //
        // The upper cut is at the inscribed Nyquist circle rather than the
        // corner of the square: modes beyond it are shorter than two cells
        // along the diagonal, so they are aliasing artefacts rather than waves,
        // and asking the fit to honour them costs accuracy on real ones.
        const double k_lo = band_low_frac * k_max;
        constexpr double kOutOfBandWeight = 0.02;

        const int ns = static_cast<int>(fit_samples);
        int orb[8][2];
        for (int zi = 0; zi <= ns; ++zi) {
            for (int xi = 0; xi <= ns; ++xi) {
                // The quadrant kx, kz in [0, k_max] represents the whole
                // resolved band, because both the true symbol and every basis
                // function are even in each axis.
                const double kx = k_max * xi / ns;
                const double kz = k_max * zi / ns;
                const double kk = std::sqrt(kx * kx + kz * kz);
                const double s  = iwave_symbol(kk, depth);

                const bool in_band = (kk >= k_lo && kk <= k_max);
                const double ref = in_band ? s : iwave_symbol(
                    (kk < k_lo) ? k_lo : k_max, depth);
                double w = 1.0 / (ref * ref);
                if (!in_band) w *= kOutOfBandWeight;

                for (int t = 1; t < m; ++t) {
                    const int a = ai[static_cast<std::size_t>(t)];
                    const int b = bi[static_cast<std::size_t>(t)];
                    const int n = orbit_of(a, b, orb);
                    double phi = 0.0;
                    for (int e = 0; e < n; ++e) {
                        phi += std::cos(kx * orb[e][0] * cell +
                                        kz * orb[e][1] * cell);
                    }
                    psi[static_cast<std::size_t>(t - 1)] = phi - n;
                }
                for (int r = 0; r < mm; ++r) {
                    const double wr = w * psi[static_cast<std::size_t>(r)];
                    atb[static_cast<std::size_t>(r)] += wr * s;
                    for (int c = 0; c <= r; ++c) {
                        ata[static_cast<std::size_t>(r) * smm +
                            static_cast<std::size_t>(c)] +=
                            wr * psi[static_cast<std::size_t>(c)];
                    }
                }
            }
        }
        // Mirror the lower triangle into the upper, and add a tiny ridge for
        // numerical safety. The ridge is many orders below the diagonal, so it
        // cannot bend the fit - it only stops an exactly-singular basis, which
        // would mean a bug, and which the PD check below would catch anyway.
        double trace = 0.0;
        for (int r = 0; r < mm; ++r) {
            trace += ata[static_cast<std::size_t>(r) * smm + static_cast<std::size_t>(r)];
        }
        const double ridge = 1e-12 * trace / mm;
        for (int r = 0; r < mm; ++r) {
            ata[static_cast<std::size_t>(r) * smm + static_cast<std::size_t>(r)] += ridge;
            for (int c = r + 1; c < mm; ++c) {
                ata[static_cast<std::size_t>(r) * smm + static_cast<std::size_t>(c)] =
                    ata[static_cast<std::size_t>(c) * smm + static_cast<std::size_t>(r)];
            }
        }
        if (!cholesky_solve(ata, atb, mm)) {
            throw std::invalid_argument(
                "ocean: iWave kernel fit is singular (degenerate basis)");
        }
        double sum_rest = 0.0;
        for (int t = 1; t < m; ++t) {
            g[static_cast<std::size_t>(t)] = atb[static_cast<std::size_t>(t - 1)];
            sum_rest += orbit_size(ai[static_cast<std::size_t>(t)],
                                   bi[static_cast<std::size_t>(t)]) *
                        g[static_cast<std::size_t>(t)];
        }
        g[0] = -sum_rest;   // the eliminated centre tap
    }

    // Expand the distinct values over their D4 orbits into the dense stencil.
    // Writing every orbit member from ONE stored value is what makes the
    // 8-fold symmetry exact rather than approximately-equal-after-rounding.
    float* taps = out.taps.data();
    {
        int orb[8][2];
        for (int t = 0; t < m; ++t) {
            const int a = ai[static_cast<std::size_t>(t)];
            const int b = bi[static_cast<std::size_t>(t)];
            const int n = orbit_of(a, b, orb);
            const float v = static_cast<float>(g[static_cast<std::size_t>(t)]);
            for (int e = 0; e < n; ++e) {
                const int i = orb[e][0], j = orb[e][1];
                taps[static_cast<std::size_t>(j + p) * static_cast<std::size_t>(width) +
                     static_cast<std::size_t>(i + p)] = v;
            }
        }
    }

    if (method == KernelMethod::Hankel) {
        // Truncation leaves a residual DC response. Remove it so that a
        // uniform water level feels exactly no force, matching the true
        // symbol's S(0) = 0. Subtracting the mean preserves D4 symmetry
        // because it is the same constant everywhere.
        double sum = 0.0;
        const std::size_t count = static_cast<std::size_t>(width) * static_cast<std::size_t>(width);
        for (std::size_t t = 0; t < count; ++t) sum += taps[t];
        const float mean = static_cast<float>(sum / static_cast<double>(count));
        for (std::size_t t = 0; t < count; ++t) taps[t] -= mean;
    }

    // Characterise what we actually built: the largest realised symbol (which
    // sets the stability limit) and the peak relative dispersion error.
    //
    // max_symbol is taken over the whole square [0, k_max]^2 including the
    // corners, NOT over the inscribed circle the fit targets. Those corner
    // modes exist on the grid whether or not they are physically meaningful,
    // and the leapfrog stability bound is set by the largest frequency the
    // state can actually contain - so leaving them out would produce a dt
    // limit that looked fine and was not.
    //
    // The error, by contrast, is reported over [band_low, k_max] only, which
    // is exactly the band the kernel claims. Quoting it below band_low would
    // measure the method's known domain limit rather than this kernel.
    constexpr int kScan = 192;
    double worst = 0.0, peak = 0.0, trough = 0.0;
    out.band_low = band_low_frac * k_max;
    for (int zi = 0; zi <= kScan; ++zi) {
        for (int xi = 0; xi <= kScan; ++xi) {
            const double kx = k_max * xi / kScan;
            const double kz = k_max * zi / kScan;
            const double kk = std::sqrt(kx * kx + kz * kz);
            const double rs = realised_symbol(out, kx, kz);
            if (rs > peak)   peak   = rs;
            if (rs < trough) trough = rs;
            if (kk >= out.band_low && kk <= k_max) {
                const double tr = iwave_symbol(kk, depth);
                if (tr > 0.0) {
                    const double e = std::fabs(rs / tr - 1.0);
                    if (e > worst) worst = e;
                }
            }
        }
    }
    out.max_symbol          = peak;
    out.min_symbol          = trough;
    out.peak_relative_error = worst;

}

}  // namespace ocean::detail

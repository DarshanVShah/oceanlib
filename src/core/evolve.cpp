#include "core/evolve.hpp"

#include "core/fft_kernel.hpp"   // for OCEAN_HAS_X86_KERNELS
#include "core/trig.hpp"

#include <cmath>

namespace ocean::detail {

void FieldSet::allocate(std::uint32_t size, bool with_velocity)
{
    n     = size;
    cells = static_cast<std::size_t>(size) * size;
    // Eight real planes normally; twelve with velocity, of which eleven carry
    // data. The twelfth is the unused imaginary half of the field that holds
    // u_z, and it is the price of packing an ODD number of real fields two to
    // a complex transform. Filling it with a real-input transform trick would
    // save half a transform out of six and cost a special case in the plan.
    const unsigned planes = with_velocity ? 12u : 8u;
    complex_fields = with_velocity ? 6 : 4;
    storage = AlignedBuffer<float>(planes * cells);
}

// ---------------------------------------------------------------------------
// Spectrum evolution
// ---------------------------------------------------------------------------

void evolve_rows_scalar(const SpectrumTables& tables, double time,
                        FieldSet& fields, std::uint32_t row_begin,
                        std::uint32_t row_end) noexcept
{
    const std::uint32_t n = tables.n;
    const double time_over_two_pi = time * kInvTwoPiD;

    float* p0_re = fields.re(0); float* p0_im = fields.im(0);
    float* p1_re = fields.re(1); float* p1_im = fields.im(1);
    float* p2_re = fields.re(2); float* p2_im = fields.im(2);
    float* p3_re = fields.re(3); float* p3_im = fields.im(3);

    for (std::uint32_t y = row_begin; y < row_end; ++y) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * n + x;

            // Phase folded into [0, 2*pi) in double before narrowing; see
            // reduced_phase. This is also why update() takes absolute time -
            // there is no accumulator to drift.
            const float phase = reduced_phase(tables.omega[i], time_over_two_pi);

            // Our own polynomial, not std::sinf/std::cosf. Two reasons: it
            // produces both values from one shared argument reduction, and it
            // is the same sequence of IEEE operations on every platform -
            // which closes the last cross-runtime determinism gap from
            // ADR-003, since libm is allowed to differ by an ULP between
            // implementations.
            float s = 0.0f, c = 0.0f;
            sincos_f32(phase, s, c);

            // h(k,t) = h0(k) e^{i w t} + conj(h0(-k)) e^{-i w t}
            //
            // Two counter-propagating wave trains at the same wavenumber. The
            // second term is what makes the field Hermitian and therefore the
            // surface real; it is also what produces standing-wave structure
            // where the two trains interfere.
            const float a_re = tables.h0_re[i];
            const float a_im = tables.h0_im[i];
            const float b_re = tables.h0c_re[i];
            const float b_im = tables.h0c_im[i];

            const float fwd_re = a_re * c - a_im * s;
            const float fwd_im = a_re * s + a_im * c;
            const float bwd_re = b_re * c + b_im * s;
            const float bwd_im = b_im * c - b_re * s;

            const float hr = fwd_re + bwd_re;
            const float hi = fwd_im + bwd_im;

            const float kx = tables.kx[i];
            const float kz = tables.kz[i];
            const float ki = tables.k_inv[i];  // 1/|k|, and exactly 0 at DC

            // Unit wavevector components. At DC these are 0, which correctly
            // kills the displacement there - a constant horizontal shift of
            // the whole ocean would be meaningless anyway.
            const float sx = kx * ki;
            const float sz = kz * ki;

            // The eight spectra, all derived from h(k,t) by multiplying by a
            // real or imaginary factor:
            //
            //   H     =  h                     (height)
            //   Dx    = -i (kx/|k|) h          (Tessendorf's chop)
            //   Dz    = -i (kz/|k|) h
            //   dh/dx =  i kx h                (slope; differentiation in
            //   dh/dz =  i kz h                 space is multiplication by ik)
            //   dDx/dx = (kx^2/|k|) h          (real: the two i factors cancel)
            //   dDz/dz = (kz^2/|k|) h
            //   dDx/dz = (kx kz/|k|) h
            //
            // Each is Hermitian, hence each transforms to a real field.
            const float h_re = hr,        h_im = hi;
            const float dx_re =  sx * hi, dx_im = -sx * hr;
            const float dz_re =  sz * hi, dz_im = -sz * hr;
            const float hx_re = -kx * hi, hx_im =  kx * hr;
            const float hz_re = -kz * hi, hz_im =  kz * hr;

            const float jxx = kx * sx;  // kx^2 / |k|
            const float jzz = kz * sz;  // kz^2 / |k|
            const float jxz = kx * sz;  // kx kz / |k|

            const float jxx_re = jxx * hr, jxx_im = jxx * hi;
            const float jzz_re = jzz * hr, jzz_im = jzz * hi;
            const float jxz_re = jxz * hr, jxz_im = jxz * hi;

            // Pack A + i*B:  re = A_re - B_im,  im = A_im + B_re.
            p0_re[i] = h_re   - dx_im;   p0_im[i] = h_im   + dx_re;
            p1_re[i] = dz_re  - hx_im;   p1_im[i] = dz_im  + hx_re;
            p2_re[i] = hz_re  - jxx_im;  p2_im[i] = hz_im  + jxx_re;
            p3_re[i] = jzz_re - jxz_im;  p3_im[i] = jzz_im + jxz_re;
        }
    }
}

void evolve_velocity_rows(const SpectrumTables& tables, double time,
                          FieldSet& fields, std::uint32_t row_begin,
                          std::uint32_t row_end) noexcept
{
    const std::uint32_t n = tables.n;
    const double time_over_two_pi = time * kInvTwoPiD;

    float* p4_re = fields.re(4); float* p4_im = fields.im(4);
    float* p5_re = fields.re(5); float* p5_im = fields.im(5);

    for (std::uint32_t y = row_begin; y < row_end; ++y) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * n + x;

            const float phase = reduced_phase(tables.omega[i], time_over_two_pi);
            float s = 0.0f, c = 0.0f;
            sincos_f32(phase, s, c);

            const float a_re = tables.h0_re[i];
            const float a_im = tables.h0_im[i];
            const float b_re = tables.h0c_re[i];
            const float b_im = tables.h0c_im[i];

            const float fwd_re = a_re * c - a_im * s;
            const float fwd_im = a_re * s + a_im * c;
            const float bwd_re = b_re * c + b_im * s;
            const float bwd_im = b_im * c - b_re * s;

            // D = A - B, the DIFFERENCE of the two counter-propagating trains.
            //
            // The sum is the surface height; the difference is what the
            // velocity potential sees, because the dynamic boundary condition
            // dphi/dt = -g*h integrates with opposite sign on the two branches
            // (they carry opposite signs of omega). Taking -i*omega*h instead -
            // the obvious move - is wrong, and gives a non-Hermitian spectrum
            // that transforms to an imaginary "velocity".
            //
            // D is anti-Hermitian, D(-k) = -conj(D(k)), which is precisely what
            // makes all three products below Hermitian and hence real fields.
            const float dr = fwd_re - bwd_re;
            const float di = fwd_im - bwd_im;

            const float w  = tables.omega[i];
            const float kx = tables.kx[i];
            const float kz = tables.kz[i];
            const float ki = tables.k_inv[i];
            const float sx = kx * ki;
            const float sz = kz * ki;

            // u_x = -omega*(kx/|k|)*D,  u_z likewise,  u_y = i*omega*D.
            const float wsx = w * sx;
            const float wsz = w * sz;
            const float ux_re = -wsx * dr, ux_im = -wsx * di;
            const float uz_re = -wsz * dr, uz_im = -wsz * di;
            const float uy_re = -w * di,   uy_im =  w * dr;

            // Pack A + i*B exactly as evolve_rows_scalar does.
            p4_re[i] = ux_re - uy_im;  p4_im[i] = ux_im + uy_re;
            p5_re[i] = uz_re;          p5_im[i] = uz_im;
        }
    }
}

void finalize_velocity_rows(const FieldSet& fields, float* velocity,
                            std::uint32_t row_begin,
                            std::uint32_t row_end) noexcept
{
    const std::uint32_t n = fields.n;
    const float* f4_re = fields.re(4); const float* f4_im = fields.im(4);
    const float* f5_re = fields.re(5);

    for (std::uint32_t y = row_begin; y < row_end; ++y) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * n + x;
            velocity[4 * i + 0] = f4_re[i];   // u_x
            velocity[4 * i + 1] = f4_im[i];   // u_y
            velocity[4 * i + 2] = f5_re[i];   // u_z
            velocity[4 * i + 3] = 0.0f;       // reserved
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

namespace {

using EvolveKernel = void (*)(const SpectrumTables&, double, FieldSet&,
                              std::uint32_t, std::uint32_t) noexcept;

EvolveKernel select_evolve_kernel() noexcept
{
#if defined(OCEAN_HAS_X86_KERNELS)
    if (detect_simd_level() == SimdLevel::Avx2) return &evolve_rows_avx2;
#endif
    return &evolve_rows_scalar;
}

}  // namespace

void evolve_rows(const SpectrumTables& tables, double time, FieldSet& fields,
                 std::uint32_t row_begin, std::uint32_t row_end) noexcept
{
    // Resolved once, on first call. Like the FFT kernels, every implementation
    // is required to be bit-identical to the scalar one, so this changes speed
    // and nothing else.
    static const EvolveKernel kernel = select_evolve_kernel();
    kernel(tables, time, fields, row_begin, row_end);
}

// ---------------------------------------------------------------------------
// Finalisation
// ---------------------------------------------------------------------------

void finalize_rows(const FieldSet& fields, float choppiness,
                   float foam_threshold, float* displacement, float* normal,
                   std::uint32_t row_begin, std::uint32_t row_end) noexcept
{
    const std::uint32_t n = fields.n;
    const float lambda = choppiness;

    const float* f0_re = fields.re(0); const float* f0_im = fields.im(0);
    const float* f1_re = fields.re(1); const float* f1_im = fields.im(1);
    const float* f2_re = fields.re(2); const float* f2_im = fields.im(2);
    const float* f3_re = fields.re(3); const float* f3_im = fields.im(3);

    // Guard a degenerate threshold so the foam ramp below never divides by 0.
    const float inv_threshold =
        (foam_threshold > 1e-6f) ? (1.0f / foam_threshold) : 0.0f;

    for (std::uint32_t y = row_begin; y < row_end; ++y) {
        for (std::uint32_t x = 0; x < n; ++x) {
            const std::size_t i = static_cast<std::size_t>(y) * n + x;

            // Unpack: real part is the first field, imaginary the second.
            const float h   = f0_re[i];
            const float d_x = f0_im[i];
            const float d_z = f1_re[i];
            const float hx  = f1_im[i];  // dh/dx
            const float hz  = f2_re[i];  // dh/dz
            const float jxx = f2_im[i];  // dDx/dx
            const float jzz = f3_re[i];  // dDz/dz
            const float jxz = f3_im[i];  // dDx/dz == dDz/dx

            // The surface point is (u + lambda*Dx, h, v + lambda*Dz).
            const float disp_x = lambda * d_x;
            const float disp_z = lambda * d_z;

            // Derivatives of that mapping.
            const float axx = 1.0f + lambda * jxx;
            const float azz = 1.0f + lambda * jzz;
            const float axz = lambda * jxz;

            // Jacobian determinant of the horizontal displacement map. It is 1
            // on flat water, drops below 1 where the surface compresses toward
            // a crest, and goes negative where the mapping folds over itself -
            // physically, where the wave is breaking.
            const float jacobian = axx * azz - axz * axz;

            // Exact normal of the DISPLACED surface, not of the heightfield.
            //
            // Parameterise the surface by (u,v). The tangents are
            //   T_u = (axx, hx, axz)
            //   T_v = (axz, hz, azz)
            // and the normal is -(T_u x T_v), negated so +Y points up. Note
            // the Y component comes out as exactly the Jacobian above, so the
            // two share the work.
            //
            // Using (-hx, 1, -hz) instead - the plain heightfield normal -
            // is the common shortcut, and it is wrong wherever choppiness is
            // doing anything: the vertex has been dragged sideways, so the
            // surface there is steeper than the height derivative alone says.
            // Shading and physics would disagree, breaking promise #2. Here it
            // costs nothing extra, because the foam term already needs these
            // same displacement gradients.
            float nx = -(hx * azz - axz * hz);
            float ny = jacobian;
            float nz = -(axx * hz - hx * axz);

            const float len2 = nx * nx + ny * ny + nz * nz;
            const float inv_len = (len2 > 0.0f) ? 1.0f / std::sqrt(len2) : 0.0f;
            nx *= inv_len;
            ny *= inv_len;
            nz *= inv_len;

            // Foam ramps in as the Jacobian falls below the threshold, and
            // saturates once the surface has actually inverted (J <= 0).
            float foam = (foam_threshold - jacobian) * inv_threshold;
            if (foam < 0.0f) foam = 0.0f;
            if (foam > 1.0f) foam = 1.0f;

            const std::size_t o = i * 4;
            displacement[o + 0] = disp_x;
            displacement[o + 1] = h;
            displacement[o + 2] = disp_z;
            displacement[o + 3] = foam;

            normal[o + 0] = nx;
            normal[o + 1] = ny;
            normal[o + 2] = nz;
            normal[o + 3] = jacobian;
        }
    }
}

}  // namespace ocean::detail

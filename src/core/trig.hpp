// oceanlib - internal header. Not part of the public API.
//
// Our own sine and cosine.
//
// TWO reasons, and the second is the important one.
//
// 1. Speed. The per-frame path needs a sin AND a cos of the same angle for
//    every grid cell - 262 144 pairs per frame at 512^2. Calling std::sinf and
//    std::cosf separately performs the argument reduction twice and cannot be
//    vectorised, because the compiler will not inline a libm call into a
//    vector loop.
//
// 2. Determinism. ADR-003 left one gap open: libm's transcendental functions
//    are permitted to differ by an ULP between implementations, so a build on
//    glibc and a build on MSVC could produce oceans that differ in the last
//    bits. A polynomial we evaluate ourselves is the same sequence of IEEE
//    multiplies and adds everywhere, so it closes that gap completely. This is
//    what makes "same seed, same waves" true ACROSS PLATFORMS and not merely
//    across runs of one binary.
//
// The algorithm is the classic Cephes single-precision reduction: fold the
// argument into [-pi/4, pi/4] by quadrant, evaluate a minimax polynomial
// there, then pick and sign the result from the quadrant. It is branch-free
// and maps directly onto SIMD.
#pragma once

#include <cmath>
#include <cstdint>

namespace ocean::detail {

// 2/pi, for finding the quadrant.
inline constexpr float kTwoOverPi = 0.636619772367581343076f;

// pi/2 split so that q * (pi/2) can be subtracted accurately.
//
// kPiOver2A is pi/2 rounded to float; kPiOver2B is what that rounding threw
// away. Subtracting in two steps (Cody-Waite reduction) keeps the reduced
// argument accurate to far more than float precision. Using a single float
// pi/2 would leak up to |q| * 4.4e-8 of error straight into the result, which
// at our magnitudes is several ULP.
inline constexpr float kPiOver2A = 1.5707963705062866f;
inline constexpr float kPiOver2B = -4.3711388286737929e-8f;

// Minimax coefficients on [-pi/4, pi/4] (Cephes sinf/cosf).
inline constexpr float kSin0 = -1.9515295891e-4f;
inline constexpr float kSin1 =  8.3321608736e-3f;
inline constexpr float kSin2 = -1.6666654611e-1f;

inline constexpr float kCos0 =  2.443315711809948e-5f;
inline constexpr float kCos1 = -1.388731625493765e-3f;
inline constexpr float kCos2 =  4.166664568298827e-2f;

inline constexpr double kTwoPiD    = 6.28318530717958647692;
inline constexpr double kInvTwoPiD = 0.159154943091895335769;

// Reduces omega*t into [0, 2*pi) and narrows to float.
//
// Done as d - floor(d) on the scaled value rather than with std::fmod. fmod is
// a libm call - tens of cycles, and impossible to inline into a vector loop -
// and it turns out to cost more here than the sine itself. floor is a single
// instruction on every target we care about, and vectorises.
//
// The arithmetic stays in double because omega*t reaches thousands of radians
// after an hour of simulated time, where float would have only ~1e-4 rad of
// resolution left. Doing the fold first means seeking to t = 10 hours is
// exactly as accurate as seeking to t = 0.1 s.
// `time_over_two_pi` is time * (1/2pi), hoisted by the caller.
//
// Hoisting is not just an optimisation here: it fixes the number and order of
// the multiplies. Writing `omega * time * kInvTwoPiD` inline would be two
// multiplies with an intermediate rounding, while the vector path would
// naturally use one multiply by a precomputed constant - and the two would
// disagree in the last bit. Bit-exactness between kernels has to be designed
// in, not hoped for.
inline float reduced_phase(float omega, double time_over_two_pi) noexcept
{
    const double d = static_cast<double>(omega) * time_over_two_pi;
    return static_cast<float>((d - std::floor(d)) * kTwoPiD);
}

// Simultaneous sine and cosine of `x`, which must already be reduced to
// roughly [-2*pi, 2*pi] (the caller does that in double, where it is exact).
//
// Computing both together is most of the win: the quadrant fold is the
// expensive part and it is shared, so a pair costs barely more than one.
inline void sincos_f32(float x, float& out_sin, float& out_cos) noexcept
{
    // Quadrant. floor(v + 0.5) rather than std::round or nearbyint: round()
    // is a libm call, and nearbyint() depends on the current rounding mode,
    // which we do not control. floor-plus-half is a fixed rule that the SIMD
    // kernels reproduce exactly.
    const float qf = std::floor(x * kTwoOverPi + 0.5f);
    const std::int32_t q = static_cast<std::int32_t>(qf);

    // Cody-Waite: subtract q*(pi/2) in two pieces.
    float y = x - qf * kPiOver2A;
    y = y - qf * kPiOver2B;

    const float z = y * y;

    // sin(y) = y + y*z*P(z)
    const float sin_y =
        y + ((kSin0 * z + kSin1) * z + kSin2) * z * y;

    // cos(y) = 1 - z/2 + z^2*Q(z)
    const float cos_y =
        1.0f - 0.5f * z + ((kCos0 * z + kCos1) * z + kCos2) * z * z;

    // Quadrant fold. For q mod 4:
    //   0: ( sin,  cos)   1: ( cos, -sin)
    //   2: (-sin, -cos)   3: (-cos,  sin)
    // `q & 3` is correct for negative q too: (-1) & 3 == 3, and quadrant -1 is
    // congruent to quadrant 3.
    const std::int32_t quadrant = q & 3;

    // Written as selects rather than a switch so the SIMD versions can mirror
    // this exactly with blends and produce bit-identical results.
    const bool swap   = (quadrant & 1) != 0;
    const float base_s = swap ? cos_y : sin_y;
    const float base_c = swap ? sin_y : cos_y;

    // sin is negative in quadrants 2 and 3; cos is negative in 1 and 2.
    const bool neg_s = (quadrant & 2) != 0;
    const bool neg_c = ((quadrant + 1) & 2) != 0;

    out_sin = neg_s ? -base_s : base_s;
    out_cos = neg_c ? -base_c : base_c;
}

}  // namespace ocean::detail

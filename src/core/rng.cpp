#include "core/rng.hpp"

#include <cmath>

namespace ocean::detail {

GaussianPair next_gaussian_pair(Pcg32& rng) noexcept
{
    const double u1 = next_open01(rng);
    const double u2 = next_open01(rng);

    const double radius = std::sqrt(-2.0 * std::log(u1));
    const double theta  = 6.283185307179586476925286766559 * u2;  // 2*pi

    return GaussianPair{static_cast<float>(radius * std::cos(theta)),
                        static_cast<float>(radius * std::sin(theta))};
}

void fill_gaussians(Pcg32& rng, float* dst, std::size_t count) noexcept
{
    std::size_t i = 0;
    for (; i + 1 < count; i += 2) {
        const GaussianPair g = next_gaussian_pair(rng);
        dst[i]     = g.a;
        dst[i + 1] = g.b;
    }
    if (i < count) {
        // Odd tail: take the first of a pair and drop the second. Wasteful by
        // one draw, but keeps `dst[i]` a function of `i` alone.
        dst[i] = next_gaussian_pair(rng).a;
    }
}

}  // namespace ocean::detail

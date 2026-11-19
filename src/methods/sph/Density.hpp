#pragma once
#include <Kokkos_Core.hpp>
#include <nexora/core/Result.hpp>
#include <nexora/core/Types.hpp>

namespace nexora::sph {
using Values = Kokkos::View<real*, Kokkos::HostSpace>;

// Recover the subtraction residual for outer-branch distances in [1, 2].
// Keeping these low bits preserves nonzero tails at the support boundary.
KOKKOS_INLINE_FUNCTION real outer_support_gap(real x, real y, real difference) {
    const real high = difference >= real{0} ? x : y;
    const real negated_low = difference >= real{0} ? -y : -x;
    const real q = Kokkos::abs(difference);
    const real residual = high >= negated_low ? (high - q) + negated_low
                                              : (negated_low - q) + high;
    return (real{2} - q) - residual;
}

// Fixed-graph validation includes the exact support boundary at distance 2.
inline bool within_support(real x, real y) {
    const real difference = x - y;
    const real q = Kokkos::abs(difference);
    return q < real{2} ||
           (q == real{2} && outer_support_gap(x, y, difference) >= real{0});
}

struct DensityInput {
    Values::const_type positions, masses;
    Kokkos::View<const Offset*, Kokkos::HostSpace> offsets;
    Kokkos::View<const LocalIndex*, Kokkos::HostSpace> indices, self_columns;
};

// The caller binds validated non-self CSR and row-to-column indices to finite particle
// fields, with positive masses and neighbors within support, and owns a disjoint trial
// output with one value per row. The host scalar reduction completes the gather on
// the supplied instance before publication or discard. The owned diagnostic uses
// ordinary double accumulation, then checks its rounded finite value against real range.
Result<real> evaluate_density(const Kokkos::Serial&, const DensityInput&, const Values& output);
} // namespace nexora::sph

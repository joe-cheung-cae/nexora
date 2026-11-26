#pragma once

#include "neighbors/NeighborView3D.hpp"
#include <nexora/core/Result.hpp>

#include <numbers>

namespace nexora::sph {

using Values = Kokkos::View<real*, Kokkos::HostSpace>;

KOKKOS_INLINE_FUNCTION bool store_positive_density(real value, real& output) {
    if (!(value > 0 && Kokkos::isfinite(value)))
        return false;
    output = value;
    return true;
}

// Dimensionless 3D M4 weight. The physical kernel is this weight divided by h cubed.
KOKKOS_INLINE_FUNCTION real m4_weight_3d(real q) {
    if (q >= 2)
        return 0;
    if (q < 1)
        return std::numbers::inv_pi_v<real> * (1 + q * q * (real{0.75} * q - real{1.5}));
    const real tail = 2 - q;
    return (real{0.25} * std::numbers::inv_pi_v<real>) * tail * tail * tail;
}

// -w'(q)/q for q < 2, including the finite limit at the origin.
KOKKOS_INLINE_FUNCTION real m4_negative_radial_derivative_over_q_inside_support_3d(real q) {
    if (q < 1)
        return std::numbers::inv_pi_v<real> * (3 - real{2.25} * q);
    const real tail = 2 - q;
    return (real{.75} * std::numbers::inv_pi_v<real>) * (tail * tail / q);
}

// Sum in sorted-neighbor order.
real density_at_slot_3d(const NeighborView3D& grid, LocalIndex slot);

// The window owns a disjoint, writable trial allocation. A failed evaluation
// may have written some rows; discard that trial before publishing a snapshot.
Status evaluate_density_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const Values& output);

} // namespace nexora::sph

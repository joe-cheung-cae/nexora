#pragma once

#include "neighbors/NeighborView3D.hpp"
#include <nexora/core/Result.hpp>

#include <cmath>
#include <limits>
#include <numbers>

namespace nexora::sph {

using Values = Kokkos::View<real*, Kokkos::HostSpace>;

// A positive double can underflow to zero when stored in a real field.
KOKKOS_INLINE_FUNCTION bool store_positive_density(double value, real& output) {
    if (!(Kokkos::abs(value) <= std::numeric_limits<real>::max()))
        return false;
    const auto rho = static_cast<real>(value);
    if (!(rho > 0))
        return false;
    output = rho;
    return true;
}

// Dimensionless 3D M4 weight. The physical kernel is this weight divided by h cubed.
KOKKOS_INLINE_FUNCTION double m4_weight_3d(double q) {
    if (q >= 2)
        return 0;
    if (q < 1.0)
        return std::numbers::inv_pi * (1.0 + q * q * (0.75 * q - 1.5));
    const double tail = 2.0 - q;
    return (0.25 * std::numbers::inv_pi) * tail * tail * tail;
}

// -w'(q)/q, including the finite limit at the origin.
KOKKOS_INLINE_FUNCTION double m4_negative_radial_derivative_over_q_inside_support_3d(double q) {
    if (q < 1)
        return std::numbers::inv_pi * (3 - 2.25 * q);
    const double tail = 2 - q;
    return (.75 * std::numbers::inv_pi) * (tail * tail / q);
}

KOKKOS_INLINE_FUNCTION double m4_negative_radial_derivative_over_q_3d(double q) {
    return q >= 2 ? 0 : m4_negative_radial_derivative_over_q_inside_support_3d(q);
}

inline double strict_m4_cutoff_squared_3d(double support, double h) {
    double cutoff = support * support;
    // Even below R², sqrt(d²)/h can round to 2. Move the exclusive bound
    // to the first such value. The computed normalized distance is monotone.
    for (auto previous = std::nextafter(cutoff, 0.); Kokkos::sqrt(previous) / h >= 2;
         previous = std::nextafter(cutoff, 0.))
        cutoff = previous;
    return cutoff;
}

// Double-precision sum in sorted-neighbor order; callers check the result before narrowing.
double density_at_slot_3d(const NeighborView3D& grid, LocalIndex slot);

// Reference density for the three-dimensional M4 kernel. Output row i
// belongs to grid.owned_slots()(i). The returned allocation is independent of
// the grid and is available on the host after the Serial reduction completes.
Result<Values> evaluate_density_3d(const Kokkos::Serial& exec, const NeighborView3D& grid);

// The window owns a disjoint, writable trial allocation. A failed evaluation
// may have written some rows; discard that trial before publishing a snapshot.
Status evaluate_density_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const Values& output);

} // namespace nexora::sph

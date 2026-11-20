#pragma once

#include "neighbors/CellGrid.hpp"

#include <numbers>

namespace nexora::sph {

using Values = Kokkos::View<real*, Kokkos::HostSpace>;

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
KOKKOS_INLINE_FUNCTION double m4_negative_radial_derivative_over_q_3d(double q) {
    if (q >= 2)
        return 0;
    if (q < 1)
        return std::numbers::inv_pi * (3 - 2.25 * q);
    const double tail = 2 - q;
    return (.75 * std::numbers::inv_pi) * (tail * tail / q);
}

// Double-precision sum in sorted-neighbor order; callers check the result before narrowing.
double density_at_slot_3d(const CellGrid& grid, LocalIndex slot);

// Reference density for the three-dimensional M4 kernel. Output row i
// belongs to grid.owned_slots()(i). The returned allocation is independent of
// the grid and is available on the host after the Serial reduction completes.
Result<Values> evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid);

// The window owns a disjoint, writable trial allocation. A failed evaluation
// may have written some rows; discard that trial before publishing a snapshot.
Status evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid, const Values& output);

} // namespace nexora::sph

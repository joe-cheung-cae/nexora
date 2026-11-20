#pragma once

#include "Density.hpp"
#include "neighbors/CellGrid.hpp"

#include <numbers>

namespace nexora::sph {

// Three-dimensional normalized cubic spline, evaluated only inside R=2.
KOKKOS_INLINE_FUNCTION double weight_3d(double distance_squared) {
    const double q = Kokkos::sqrt(distance_squared);
    if (q < 1.0)
        return std::numbers::inv_pi * (1.0 + q * q * (0.75 * q - 1.5));
    const double tail = 2.0 - q;
    return (0.25 * std::numbers::inv_pi) * tail * tail * tail;
}

// Double-precision sum in sorted-neighbor order; callers check the result before narrowing.
double density_at_slot_3d(const CellGrid& grid, LocalIndex slot);

// Reference density for the fixed h=1 three-dimensional M4 slice. Output row i
// belongs to grid.owned_slots()(i). The returned allocation is independent of
// the grid and is available on the host after the Serial reduction completes.
Result<Values> evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid);

// The window owns a disjoint, writable trial allocation. A failed evaluation
// may have written some rows; discard that trial before publishing a snapshot.
Status evaluate_density_3d(const Kokkos::Serial& exec, const CellGrid& grid, const Values& output);

} // namespace nexora::sph

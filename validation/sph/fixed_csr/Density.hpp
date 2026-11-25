#pragma once
#include <Kokkos_Core.hpp>
#include <nexora/core/Result.hpp>
#include <nexora/core/Types.hpp>

namespace nexora::sph_fixture {
using Values = Kokkos::View<real*, Kokkos::HostSpace>;

// Fixed-graph validation includes the exact support boundary at distance 2.
inline bool within_support(real x, real y) {
    return Kokkos::abs(x - y) <= real{2};
}

struct DensityInput {
    Values::const_type positions, masses;
    Kokkos::View<const Offset*, Kokkos::HostSpace> offsets;
    Kokkos::View<const LocalIndex*, Kokkos::HostSpace> indices, self_columns;
};

// The caller binds validated non-self CSR and row-to-column indices to finite particle
// fields, with positive masses and neighbors within support, and owns a disjoint
// output with one value per row. The host scalar reduction completes the gather on
// the supplied instance before publication or discard. The owned diagnostic uses
// ordinary double accumulation, then checks its rounded finite value against real range.
Result<real> evaluate_density(const Kokkos::Serial&, const DensityInput&, const Values& output);
} // namespace nexora::sph_fixture

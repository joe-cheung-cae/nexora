#pragma once

#include "Density3D.hpp"
#include "TaitEos.hpp"

#include <array>

namespace nexora::sph {

using Acceleration3D = Kokkos::View<real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;
using WallAcceleration3D = Kokkos::View<double* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Zero owned rows use no workspace. Otherwise multiply by the number of source slots.
std::size_t pressure_workspace_bytes_per_source(bool has_ghosts, bool adami_walls = false);

// Owned density is the completed density of this geometry. The three fields must be disjoint.
// Ghost reconstruction requires the source halo to cover each contributing ghost's full support.
// Outputs are disposable trial storage if any stage fails.
Status evaluate_pressure_3d(const Kokkos::Serial&, const CellGrid&, const TaitEos&,
                            const Values::const_type& owned_density, const Values& owned_pressure,
                            const Acceleration3D& owned_acceleration);

// Every ghost is an Adami wall. Its acceleration uses the same stage's sorted slot order.
// Wall acceleration is read only if this geometry has both owned and wall sources.
Status evaluate_pressure_3d(const Kokkos::Serial&, const CellGrid&, const TaitEos&,
                            const Values::const_type& owned_density, const Values& owned_pressure,
                            const Acceleration3D& owned_acceleration,
                            const WallAcceleration3D::const_type& wall_acceleration,
                            const std::array<double, 3>& gravity);

} // namespace nexora::sph

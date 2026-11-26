#pragma once
#include "Verlet3D.hpp"

#include <functional>

namespace nexora::sph {

using DensityRate3D = Kokkos::View<real*, Kokkos::HostSpace>;

using StageVelocityReady3D = Kokkos::View<std::uint8_t*, Kokkos::HostSpace>;
using WriteWallVelocity3D = std::function<bool(LocalIndex)>;

// The WCSPH step provides disjoint views at the grid's time. Fluid velocity and rates
// follow owned rows. The slot map can be empty when there are no walls.
// Missing wall callback gives zero velocity. The callback fills sorted wall slots on demand.
Status evaluate_density_rate_3d(const Kokkos::Serial&, const NeighborView3D&,
    const Velocity3D::const_type& owned_velocity, const NeighborView3D::Indices& slot_to_owned_row,
    const DensityRate3D&, const Velocity3D& wall_velocity = {},
    const StageVelocityReady3D& wall_ready = {}, const WriteWallVelocity3D& write_wall = {});

} // namespace nexora::sph

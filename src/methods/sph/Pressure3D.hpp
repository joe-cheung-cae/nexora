#pragma once

#include "Density3D.hpp"
#include "TaitEos.hpp"

#include <array>
#include <functional>

namespace nexora::sph {

using Acceleration3D = Kokkos::View<real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;
using WallAcceleration3D = Kokkos::View<double* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Zero owned rows use no workspace. Otherwise multiply by the number of source slots.
std::size_t step_pressure_workspace_bytes_per_source(bool has_walls);

// Owned density is the completed density of this geometry. The three fields must be disjoint.
// Ghost reconstruction requires the source halo to cover each contributing ghost's full support.
// Outputs are disposable trial storage if any stage fails.
Status evaluate_pressure_3d(const Kokkos::Serial&, const NeighborView3D&, const TaitEos&,
                            const Values::const_type& owned_density, const Values& owned_pressure,
                            const Acceleration3D& owned_acceleration);

// Every ghost is an Adami wall. Its acceleration uses the same stage's sorted slot order.
// Wall acceleration and gravity enter the reconstruction only for contributing walls.
// Wall acceleration must be disjoint from owned pressure and may overlap owned acceleration.
Status evaluate_pressure_3d(const Kokkos::Serial&, const NeighborView3D&, const TaitEos&,
                            const Values::const_type& owned_density, const Values& owned_pressure,
                            const Acceleration3D& owned_acceleration,
                            const WallAcceleration3D::const_type& wall_acceleration,
                            const std::array<double, 3>& gravity);

// WCSPH has validated the fixed EOS and gravity at initialization. Its field grants
// cover disjoint owned outputs, its corrected density writer guarantees positive finite values,
// and every ghost belongs to an Adami group. Its wall motion is independent of the
// acceleration output, so a wall source may be sampled during the Serial force loop.
using WallAccelerationAt3D = std::function<std::array<double, 3>(LocalIndex)>;
Status evaluate_step_pressure_3d(const Kokkos::Serial&, const NeighborView3D&, const TaitEos&,
                                 const Values::const_type& density, const Values& pressure,
                                 const Acceleration3D&, const WallAccelerationAt3D&,
                                 const std::array<double, 3>& gravity, bool initial_density);

} // namespace nexora::sph

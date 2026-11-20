#pragma once

#include "Density3D.hpp"

#include <array>

namespace nexora::sph {

struct WcsphParameters {
    double reference_density{};
    double sound_speed{};
    double gamma{};
    Status validate() const;
};

using Acceleration3D = Kokkos::View<real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace>;

// Zero owned rows use no workspace. Otherwise multiply by the number of source slots.
std::size_t pressure_workspace_bytes_per_source(bool has_ghosts, bool fixed_walls = false);

// Owned density is the completed density of this geometry. The three fields must be disjoint.
// Ghost reconstruction requires the source halo to cover each contributing ghost's full support.
// With wall_gravity set, every ghost is a stationary Adami wall source.
// Outputs are disposable trial storage if any stage fails.
Status evaluate_pressure_3d(const Kokkos::Serial&, const CellGrid&, const WcsphParameters&,
                            const Values::const_type& owned_density, const Values& owned_pressure,
                            const Acceleration3D& owned_acceleration,
                            const std::array<double, 3>* wall_gravity = nullptr);

} // namespace nexora::sph

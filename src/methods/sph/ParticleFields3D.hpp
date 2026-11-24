#pragma once

#include <Kokkos_Core.hpp>
#include <nexora/core/Types.hpp>

#include <cstdint>

namespace nexora::sph {
struct ParticleFields3D {
    Kokkos::View<const real* [3], Kokkos::LayoutLeft, Kokkos::HostSpace> position;
    Kokkos::View<const real*, Kokkos::HostSpace> mass;
    Kokkos::View<const std::uint64_t*, Kokkos::HostSpace> ids;
    Kokkos::View<const std::uint8_t*, Kokkos::HostSpace> ghost;
};
} // namespace nexora::sph

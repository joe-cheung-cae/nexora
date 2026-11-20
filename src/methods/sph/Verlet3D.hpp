#pragma once

#include "Pressure3D.hpp"

#include <array>

namespace nexora::sph {

using Velocity3D = Acceleration3D;

using DensityRate3D = Kokkos::View<double*, Kokkos::HostSpace>;

// Velocity follows sorted source slots. Ghost slots represent stationary fixed walls.
// Output rates follow owned rows.
Status continuity_rate_3d(const Kokkos::Serial&, const CellGrid&,
                          const Velocity3D::const_type& velocity, const DensityRate3D& rate);

struct WcsphVerletParameters {
    WcsphParameters eos;
    double cfl{};
    std::array<double, 3> gravity{};
    double smoothing_length{1};
    Status validate() const;
};

// Velocity and acceleration follow owned fluid rows. Staged outputs follow accepted sorted
// source slots. Explicit fixed walls retain their positions. Drift uses real-rounded half velocity.
Status kick_drift_3d(const Kokkos::Serial&, const CellGrid& accepted,
                     const Velocity3D::const_type& velocity,
                     const Acceleration3D::const_type& acceleration, double dt,
                     const Velocity3D& staged_position, const Velocity3D& half_velocity,
                     const std::array<double, 3>& gravity = {}, bool fixed_walls = false);

// The refreshed grid was built from staged_position. Its sorted_to_input permutation maps
// each final velocity slot to its corresponding staged half velocity.
Status finish_kick_3d(const Kokkos::Serial&, const CellGrid& refreshed,
                      const Velocity3D::const_type& half_velocity,
                      const Acceleration3D::const_type& refreshed_acceleration, double dt,
                      const Velocity3D& velocity, const std::array<double, 3>& gravity = {},
                      bool fixed_walls = false);

// Maximum acoustic step for constant h. Input is accepted owned velocity. Empty and stationary
// fields give C*h/c0. C is an explicit safety factor in (0,1].
Result<double> cfl_step_3d(const Kokkos::Serial&, const Velocity3D::const_type& accepted_velocity,
                           double sound_speed, double cfl_coefficient, double smoothing_length = 1);

} // namespace nexora::sph

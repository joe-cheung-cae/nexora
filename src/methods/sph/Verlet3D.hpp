#pragma once

#include "Pressure3D.hpp"
#include "neighbors/CellGrid.hpp"

#include <array>

namespace nexora::sph {

using Velocity3D = Acceleration3D;

struct WcsphVerletParameters {
    WcsphParameters eos;
    real cfl{};
    std::array<real, 3> gravity{};
    real smoothing_length{1};
    Status validate() const;
};

// Velocity and acceleration follow owned fluid rows. Staged outputs follow accepted sorted
// source slots. Only owned slots are written. Drift uses real-rounded half velocity.
Status kick_drift_3d(const Kokkos::Serial&, const CellGrid& accepted,
                     const Velocity3D::const_type& velocity,
                     const Acceleration3D::const_type& acceleration, real dt,
                     const Velocity3D& staged_position, const Velocity3D& half_velocity,
                     const std::array<real, 3>& gravity = {});

// Velocity and acceleration follow owned fluid rows. The participant admits dt,
// validates the fixed parameters and supplies disjoint storage.
namespace verlet_step {
// Staged outputs follow accepted sorted source slots. Only owned slots are written.
// Build/validate the grid from staged_position before consuming either output.
// Grid admission rejects a nonfinite half velocity propagated through the drift.
void kick_drift(const Kokkos::Serial&, const CellGrid&, const Velocity3D::const_type& velocity,
                  const Acceleration3D::const_type& acceleration, real dt,
                  const Velocity3D& staged_position, const Velocity3D& half_velocity,
                  const std::array<real, 3>& gravity);

// Both writers reduce the maximum squared speed of the stored real velocities.
// The refreshed grid maps owned slots back to their drifted half velocities.
Result<real> capture_velocity(const Kokkos::Serial&, const NeighborView3D::Indices& row_to_input,
                                const Velocity3D::const_type& input, const Velocity3D& output);
Result<real> finish_kick(const Kokkos::Serial&, const CellGrid&, const Velocity3D::const_type& half,
                           const Acceleration3D::const_type& acceleration, real dt,
                           const Velocity3D& velocity, const std::array<real, 3>& gravity);
// Empty and stationary fields give C*h/c0.
Result<double> acoustic_step(real maximum_speed_squared, real sound_speed,
                             real cfl_coefficient, real smoothing_length);
} // namespace verlet_step

} // namespace nexora::sph

#pragma once

#include "Pressure3D.hpp"
#include "neighbors/CellGrid.hpp"

#include <array>

namespace nexora::sph {

using Velocity3D = Acceleration3D;

struct WcsphVerletParameters {
    WcsphParameters eos;
    double cfl{};
    std::array<double, 3> gravity{};
    double smoothing_length{1};
    Status validate() const;
};

// Velocity and acceleration follow owned fluid rows. Staged outputs follow accepted sorted
// source slots. Only owned slots are written. Drift uses real-rounded half velocity.
Status kick_drift_3d(const Kokkos::Serial&, const CellGrid& accepted,
                     const Velocity3D::const_type& velocity,
                     const Acceleration3D::const_type& acceleration, double dt,
                     const Velocity3D& staged_position, const Velocity3D& half_velocity,
                     const std::array<double, 3>& gravity = {});

// The refreshed grid was built from staged_position. Its sorted_to_input permutation maps
// each final velocity slot to its corresponding staged half velocity.
Status finish_kick_3d(const Kokkos::Serial&, const CellGrid& refreshed,
                      const Velocity3D::const_type& half_velocity,
                      const Acceleration3D::const_type& refreshed_acceleration, double dt,
                      const Velocity3D& velocity, const std::array<double, 3>& gravity = {});

// Maximum acoustic step for constant h. A nonfinite owned velocity yields no usable step.
// Empty and stationary fields give C*h/c0. C is an explicit safety factor in (0,1].
Result<double> cfl_step_3d(const Kokkos::Serial&, const Velocity3D::const_type& velocity,
                           double sound_speed, double cfl_coefficient, double smoothing_length = 1);

// The participant admits dt and validates the fixed parameters, and its workspace and
// field grants supply disjoint storage with the extents described above.
namespace verlet_step {
// Build/validate the grid from staged_position before consuming either output.
// In double storage, this coordinate admission also rejects a nonfinite half
// velocity propagated through the drift. Float narrowing is checked here.
Status kick_drift(const Kokkos::Serial&, const CellGrid&, const Velocity3D::const_type& velocity,
                  const Acceleration3D::const_type& acceleration, double dt,
                  const Velocity3D& staged_position, const Velocity3D& half_velocity,
                  const std::array<double, 3>& gravity);

// Both writers reduce the maximum squared speed of the stored real velocities.
Result<double> capture_velocity(const Kokkos::Serial&, const NeighborView3D::Indices& row_to_input,
                                const Velocity3D::const_type& input, const Velocity3D& output);
Result<double> finish_kick(const Kokkos::Serial&, const CellGrid&, const Velocity3D::const_type& half,
                           const Acceleration3D::const_type& acceleration, double dt,
                           const Velocity3D& velocity, const std::array<double, 3>& gravity);
Result<double> acoustic_step(double maximum_speed_squared, double sound_speed,
                             double cfl_coefficient, double smoothing_length);
} // namespace verlet_step

} // namespace nexora::sph

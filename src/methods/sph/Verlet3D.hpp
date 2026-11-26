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

namespace central_difference_step {
enum class TimeScheme3D { startup, centered, adams_bashforth };
struct TimeWeights3D {
    TimeScheme3D scheme;
    real dt;
    real previous_dt;
};

// The participant admits dt and supplies disjoint storage. Position output follows
// the accepted sorted slots. Grid admission validates the quadratic drift.
void drift_positions(const Kokkos::Serial&, const CellGrid&, const Velocity3D::const_type& velocity,
                     const Acceleration3D::const_type& pressure_acceleration, real dt,
                     const Velocity3D& position, const std::array<real, 3>& gravity);

// Each field follows its own time layer's owned rows. An empty row map is the
// identity. Startup does not read previous fields.
Result<real> advance_velocity(const Kokkos::Serial&,
    const NeighborView3D::Indices& next_to_current, const NeighborView3D::Indices& current_to_previous,
    const Velocity3D::const_type& velocity, const Acceleration3D::const_type& pressure_acceleration,
    const Velocity3D::const_type& previous_velocity,
    const Acceleration3D::const_type& previous_pressure_acceleration,
    TimeWeights3D, const Velocity3D& next_velocity, const std::array<real, 3>& gravity);

Status advance_density(const Kokkos::Serial&,
    const NeighborView3D::Indices& next_to_current, const NeighborView3D::Indices& current_to_previous,
    const Values::const_type& density, const Values::const_type& density_rate,
    const Values::const_type& previous_density, const Values::const_type& previous_density_rate,
    TimeWeights3D, const Values& next_density);

// Both writers reduce the maximum squared speed of the stored real velocities.
Result<real> capture_velocity(const Kokkos::Serial&, const NeighborView3D::Indices& row_to_input,
                                const Velocity3D::const_type& input, const Velocity3D& output);
// Empty and stationary fields give C*h/c0.
Result<double> acoustic_step(real maximum_speed_squared, real sound_speed,
                             real cfl_coefficient, real smoothing_length);
} // namespace central_difference_step

} // namespace nexora::sph

#include "Continuity3D.hpp"

#include <array>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

std::array<real, 3> read_components(const Velocity3D::const_type& velocity, std::size_t row) {
    return {velocity(row, 0), velocity(row, 1), velocity(row, 2)};
}
} // namespace

Status evaluate_density_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D::const_type& owned_velocity, const NeighborView3D::Indices& slot_to_owned_row,
    const DensityRate3D& rate, const Velocity3D& wall_velocity, const StageVelocityReady3D& wall_ready,
    const WriteWallVelocity3D& write_wall) {
    const auto rows = grid.owned_slots().extent(0);
    if (!rows)
        return {};
    const real h = grid.smoothing_length();
    const real inverse_h5 = 1 / (h * h * h * h * h);
    const auto& fields = grid.fields();
    if (write_wall)
        Kokkos::deep_copy(exec, wall_ready, std::uint8_t{0});
    const Velocity3D::const_type sampled_wall_velocity = wall_velocity;
    bool wall_failed = false;
    const auto read_velocity = [&](LocalIndex slot, std::array<real, 3>& velocity) {
        if (!slot_to_owned_row.extent(0)) {
            velocity = read_components(owned_velocity, slot);
        } else if (!fields.ghost(slot)) {
            velocity = read_components(owned_velocity, slot_to_owned_row(slot));
        } else if (!write_wall) {
            velocity = {};
        } else {
            // Serial reuses successful samples immediately, including across owned rows.
            if (!wall_ready(slot)) {
                if (wall_failed || !write_wall(slot)) {
                    wall_failed = true;
                    return false;
                }
                wall_ready(slot) = 1;
            }
            velocity = read_components(sampled_wall_velocity, slot);
        }
        return true;
    };
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.continuity.evaluate_density_rate", Policy(exec, 0, rows),
        [&](std::size_t row, std::size_t& errors) {
            const auto center_velocity = read_components(owned_velocity, row);
            real value = 0;
            grid.for_each_neighbor_with_displacement(row,
                [&](LocalIndex slot, real d2, real dx, real dy, real dz) {
                    const real q = Kokkos::sqrt(d2) / h;
                    if (q == 0 || q >= 2)
                        return;
                    std::array<real, 3> neighbor_velocity;
                    if (!read_velocity(slot, neighbor_velocity))
                        return;
                    const std::array displacement{dx, dy, dz};
                    real dot = 0;
                    for (int axis = 0; axis < 3; ++axis)
                        dot += (center_velocity[axis] - neighbor_velocity[axis]) * displacement[axis];
                    const real gradient = m4_negative_radial_derivative_over_q_inside_support_3d(q);
                    value += (dot * fields.mass(slot)) * gradient * inverse_h5;
                });
            rate(row) = value;
            errors += !Kokkos::isfinite(value);
        }, invalid);
    if (wall_failed)
        return {StatusCode::numerical_failure, "unrepresentable WCSPH moving wall velocity"};
    if (invalid)
        return {StatusCode::numerical_failure, "WCSPH density rate exceeds real range"};
    return {};
}

} // namespace nexora::sph

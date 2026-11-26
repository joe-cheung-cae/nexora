#include "Continuity3D.hpp"

#include <array>
#include <type_traits>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

template <bool all_slots = false, class Prepare = std::nullptr_t>
real rate_at_row(const NeighborView3D& grid, const Velocity3D::const_type& velocity,
                   std::size_t row, real h, real inverse_h5, const Prepare& prepare = nullptr) {
    const auto& sources = grid.fields();
    const auto i = grid.owned_slots()(row);
    real sum = 0;
    [[maybe_unused]] bool center_ready = false;
    [[maybe_unused]] std::array<real, 3> center_velocity;
    grid.for_each_neighbor_with_displacement(row, [&](LocalIndex j, real d2, real dx, real dy, real dz) {
        const real q = Kokkos::sqrt(d2) / h;
        if (q == 0 || q >= 2)
            return;
        if constexpr (all_slots) {
            if (!center_ready) {
                center_ready = prepare(i);
                if (!center_ready)
                    return;
                // A ready center is immutable for the remaining Serial row.
                // Neighbor preparation writes other slots, so keep these values
                // after the first successful materialization actually needed.
                for (int axis = 0; axis < 3; ++axis)
                    center_velocity[axis] = velocity(i, axis);
            }
            if (!prepare(j))
                return;
        } else if constexpr (!std::is_same_v<Prepare, std::nullptr_t>) {
            if (sources.ghost(j) && !prepare(j))
                return;
        }
        real dot = 0;
        const std::array displacement{dx, dy, dz};
        for (int axis = 0; axis < 3; ++axis) {
            const real vj = velocity(j, axis);
            if constexpr (all_slots)
                dot += (center_velocity[axis] - vj) * displacement[axis];
            else
                dot += (velocity(i, axis) - vj) * displacement[axis];
        }
        const real gradient = m4_negative_radial_derivative_over_q_inside_support_3d(q);
        sum += (dot * sources.mass(j)) * gradient * inverse_h5;
    });
    return sum;
}

template <bool all_slots, class Ready, class Write>
Status rate_with_materialized_velocity(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D& velocity, const DensityRate3D& rate, const Ready& ready,
    const Write& write) {
    const real h = grid.smoothing_length();
    const real inverse_h5 = 1 / (h * h * h * h * h);
    Kokkos::deep_copy(exec, ready, typename Ready::value_type{});
    // Serial visits the rows in one execution space. Each ready slot can be
    // reused immediately by later rows in this stage.
    bool wall_failed = false;
    const auto prepare = [&](LocalIndex slot) {
        if (ready(slot))
            return true;
        if (wall_failed)
            return false;
        if (!write(slot)) {
            wall_failed = true;
            return false;
        }
        ready(slot) = 1;
        return true;
    };
    const Velocity3D::const_type source_velocity = velocity;
    constexpr auto label = all_slots ? "sph.continuity.predicted_rate" : "sph.continuity.wall_rate";
    Kokkos::parallel_for(label, Policy(exec, 0, grid.owned_slots().extent(0)), [&](std::size_t row) {
        rate(row) = rate_at_row<all_slots>(grid, source_velocity, row, h, inverse_h5, prepare);
    });
    if (wall_failed)
        return {StatusCode::numerical_failure, "unrepresentable WCSPH moving wall velocity"};
    return {};
}

template <class Ready, class WriteWall>
Status rate_with_prediction(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D& velocity, const DensityRate3D& rate, const Ready& ready,
    const ContinuityPredictor3D& predictor, const NeighborView3D::Indices& new_to_old_slot,
    const NeighborView3D::Indices& old_row, const WriteWall& write_wall) {
    return rate_with_materialized_velocity<true>(exec, grid, velocity, rate, ready, [&](LocalIndex slot) {
        if constexpr (!std::is_same_v<WriteWall, std::nullptr_t>) {
            if (grid.fields().ghost(slot))
                return write_wall(slot);
        }
        predictor.write(old_row(new_to_old_slot(slot)), slot, velocity);
        return true;
    });
}

} // namespace

void stage_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                              const Velocity3D::const_type& velocity, const DensityRate3D& rate) {
    const real h = grid.smoothing_length();
    const real inverse_h5 = 1 / (h * h * h * h * h);
    Kokkos::parallel_for("sph.continuity.rate", Policy(exec, 0, grid.owned_slots().extent(0)),
        [&](std::size_t row) { rate(row) = rate_at_row(grid, velocity, row, h, inverse_h5); });
}

Status stage_wall_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                                    const Velocity3D& velocity, const DensityRate3D& rate,
                                    const StageVelocityReady3D& ready, const WriteWallVelocity3D& write) {
    return rate_with_materialized_velocity<false>(exec, grid, velocity, rate, ready, write);
}

Status stage_predicted_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D& velocity, const DensityRate3D& rate, const StageVelocityReady3D& ready,
    const ContinuityPredictor3D& predictor, const NeighborView3D::Indices& new_to_old_slot,
    const NeighborView3D::Indices& old_row, const WriteWallVelocity3D& write_wall) {
    return rate_with_prediction(exec, grid, velocity, rate, ready, predictor, new_to_old_slot, old_row, write_wall);
}

Status stage_predicted_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D& velocity, const DensityRate3D& rate, const Values& correction_scratch,
    const ContinuityPredictor3D& predictor, const NeighborView3D::Indices& new_to_old_slot,
    const NeighborView3D::Indices& old_row) {
    return rate_with_prediction(exec, grid, velocity, rate, correction_scratch, predictor,
                               new_to_old_slot, old_row, nullptr);
}

Status correct_density_3d(const Kokkos::Serial& exec, const NeighborView3D::Indices& new_owned,
                          const NeighborView3D::Indices& new_to_old_slot,
                          const NeighborView3D::Indices& old_row, const Values::const_type& density,
                          const DensityRate3D& rate_n, const DensityRate3D& rate_star,
                          real dt, const Values& corrected) {
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.continuity.correct_density", Policy(exec, 0, new_owned.extent(0)),
        [&](std::size_t r, std::size_t& errors) {
            const auto r0 = old_row(new_to_old_slot(new_owned(r)));
            const real rho = Kokkos::fma(real{.5} * dt, rate_n(r0) + rate_star(r), density(r0));
            errors += !store_positive_density(rho, corrected(r));
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "corrected owned WCSPH density is outside positive real range"};
    return {};
}

} // namespace nexora::sph

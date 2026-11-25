#include "Continuity3D.hpp"

#include <array>
#include <type_traits>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;
template <class A, class B> bool overlaps(const A& a, const B& b) {
    if (!a.span() || !b.span())
        return false;
    const auto before = std::less<const void*>{};
    return before(a.data(), b.data() + b.span()) && before(b.data(), a.data() + a.span());
}

template <class... Views> bool valid_fields(std::size_t n, const Views&... views) {
    return ((views.extent(0) == n && (!n || views.data() != nullptr)) && ...);
}

double continuity_cutoff_squared(const NeighborView3D& grid) {
    const double h = grid.smoothing_length(), support = grid.support_radius();
    const double squared = support * support;
    // An accepted neighbor at rounded R² is active when its normalized
    // distance still rounds below 2, for example at a subnormal R².
    if (Kokkos::sqrt(squared) / h < 2)
        return std::nextafter(squared, std::numeric_limits<double>::infinity());
    return strict_m4_cutoff_squared_3d(support, h);
}

template <bool step = false, bool all_slots = false, class Prepare = std::nullptr_t>
double rate_at_row(const NeighborView3D& grid, const Velocity3D::const_type& velocity,
                   std::size_t row, double h, double inverse_h5, double cutoff_squared = 0,
                   const Prepare& prepare = nullptr) {
    const auto& sources = grid.fields();
    const auto i = grid.owned_slots()(row);
    double sum = 0;
    [[maybe_unused]] bool center_ready = false;
    [[maybe_unused]] std::array<double, 3> center_velocity;
    grid.for_each_neighbor_with_displacement(row, [&](LocalIndex j, double d2, double dx, double dy, double dz) {
        if constexpr (step) {
            if (d2 >= cutoff_squared)
                return;
            // Coincident positions contribute zero. A tiny nonzero displacement
            // may also square to zero, but its velocity still participates.
            if (d2 == 0 && dx == 0 && dy == 0 && dz == 0)
                return;
        }
        const double q = Kokkos::sqrt(d2) / h;
        const double gradient = step ? m4_negative_radial_derivative_over_q_inside_support_3d(q)
                                     : m4_negative_radial_derivative_over_q_3d(q);
        if constexpr (!step) {
            if (gradient == 0)
                return;
        }
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
        double dot = 0;
        const std::array displacement{dx, dy, dz};
        for (int axis = 0; axis < 3; ++axis) {
            const double vj = velocity(j, axis);
            if constexpr (all_slots)
                dot += (center_velocity[axis] - vj) * displacement[axis];
            else
                dot += (double(velocity(i, axis)) - vj) * displacement[axis];
        }
        sum += (dot * sources.mass(j)) * gradient * inverse_h5;
    });
    return sum;
}

enum class VelocityFailure { none, wall, prediction };

template <bool all_slots, class Ready, class Write>
Status rate_with_materialized_velocity(const Kokkos::Serial& exec, const NeighborView3D& grid,
    const Velocity3D& velocity, const DensityRate3D& rate, const Ready& ready,
    const Write& write) {
    const double h = grid.smoothing_length();
    const double cutoff = continuity_cutoff_squared(grid);
    const double inverse_h5 = 1 / (h * h * h * h * h);
    Kokkos::deep_copy(exec, ready, typename Ready::value_type{});
    // Serial visits the rows in one execution space. Each ready slot can be
    // reused immediately by later rows in this stage.
    auto failure = VelocityFailure::none;
    const auto prepare = [&](LocalIndex slot) {
        if (ready(slot))
            return true;
        if (failure != VelocityFailure::none)
            return false;
        failure = write(slot);
        if (failure != VelocityFailure::none)
            return false;
        ready(slot) = 1;
        return true;
    };
    const Velocity3D::const_type source_velocity = velocity;
    constexpr auto label = all_slots ? "sph.continuity.predicted_rate" : "sph.continuity.wall_rate";
    Kokkos::parallel_for(label, Policy(exec, 0, grid.owned_slots().extent(0)), [&](std::size_t row) {
        rate(row) = rate_at_row<true, all_slots>(grid, source_velocity, row, h, inverse_h5, cutoff, prepare);
    });
    if (failure == VelocityFailure::wall)
        return {StatusCode::numerical_failure, "unrepresentable WCSPH moving wall velocity"};
    if (failure == VelocityFailure::prediction)
        return {StatusCode::numerical_failure, "invalid predicted WCSPH velocity"};
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
                return write_wall(slot) ? VelocityFailure::none : VelocityFailure::wall;
        }
        return predictor.write<false>(old_row(new_to_old_slot(slot)), slot, velocity)
            ? VelocityFailure::none : VelocityFailure::prediction;
    });
}

} // namespace

Status continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                          const Velocity3D::const_type& velocity, const DensityRate3D& rate) {
    const auto& sources = grid.fields();
    const auto owned = grid.owned_slots();
    if (!valid_fields(sources.ids.extent(0), velocity) || !valid_fields(owned.extent(0), rate) ||
        overlaps(rate, velocity) || overlaps(rate, sources.mass) || overlaps(rate, sources.position))
        return {StatusCode::invalid_argument, "continuity fields have invalid extents, storage or output overlap"};
    const double h = grid.smoothing_length();
    const double inverse_h5 = 1 / (h * h * h * h * h);
    std::size_t invalid{};
    Kokkos::parallel_reduce(
        "sph.continuity.rate", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t row, std::size_t& errors) {
            rate(row) = rate_at_row(grid, velocity, row, h, inverse_h5);
            errors += !Kokkos::isfinite(rate(row));
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "nonfinite SPH continuity rate"};
    return {};
}

void stage_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                              const Velocity3D::const_type& velocity, const DensityRate3D& rate) {
    const double h = grid.smoothing_length();
    const double cutoff = continuity_cutoff_squared(grid);
    const double inverse_h5 = 1 / (h * h * h * h * h);
    Kokkos::parallel_for("sph.continuity.rate", Policy(exec, 0, grid.owned_slots().extent(0)),
        [&](std::size_t row) { rate(row) = rate_at_row<true>(grid, velocity, row, h, inverse_h5, cutoff); });
}

Status stage_wall_continuity_rate_3d(const Kokkos::Serial& exec, const NeighborView3D& grid,
                                    const Velocity3D& velocity, const DensityRate3D& rate,
                                    const StageVelocityReady3D& ready, const WriteWallVelocity3D& write) {
    return rate_with_materialized_velocity<false>(exec, grid, velocity, rate, ready, [&](LocalIndex slot) {
        return write(slot) ? VelocityFailure::none : VelocityFailure::wall;
    });
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
                          double dt, const Values& corrected) {
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.continuity.correct_density", Policy(exec, 0, new_owned.extent(0)),
        [&](std::size_t r, std::size_t& errors) {
            const auto r0 = old_row(new_to_old_slot(new_owned(r)));
            const double rho = Kokkos::fma(.5 * dt, rate_n(r0) + rate_star(r), double(density(r0)));
            errors += !store_positive_density(rho, corrected(r));
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "corrected owned WCSPH density is outside positive real range"};
    return {};
}

} // namespace nexora::sph

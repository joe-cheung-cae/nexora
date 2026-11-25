#include "Pressure3D.hpp"

#include "AdamiWallPressure3D.hpp"

#include <array>
#include <functional>
#include <limits>
#include <optional>

namespace nexora::sph {
namespace {

using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

template <class A, class B> bool overlaps(const A& a, const B& b) {
    if (!a.span() || !b.span())
        return false;
    const auto before = std::less<const real*>{};
    return before(a.data(), b.data() + b.span()) && before(b.data(), a.data() + a.span());
}

bool fits_real(double value) {
    return Kokkos::abs(value) <= std::numeric_limits<real>::max();
}

bool prepare_source(const TaitEos& eos, real rho, real& pressure, double& coefficient) {
    const double raw = eos.pressure(rho);
    if (!fits_real(raw))
        return false;
    pressure = static_cast<real>(raw);
    coefficient = double(pressure) / (double(rho) * rho);
    return true;
}

std::optional<double> ghost_source_coefficient(const NeighborView3D& grid, LocalIndex slot,
                                                const TaitEos& eos) {
    const double density = density_at_slot_3d(grid, slot);
    real pressure{};
    double coefficient{};
    real rho{};
    if (!store_positive_density(density, rho) || !prepare_source(eos, rho, pressure, coefficient))
        return std::nullopt;
    return coefficient;
}

bool contributes(double distance_squared, double cutoff_squared) {
    return distance_squared > 0 && distance_squared < cutoff_squared;
}

template <class Prepare>
std::array<double, 3>
acceleration_at_slot(const NeighborView3D& grid, LocalIndex i,
                     const Kokkos::View<const double*, Kokkos::HostSpace>& force_coefficient,
                     double cutoff_squared, double inverse_h5, const Prepare& prepare) {
    const auto& position = grid.fields().position;
    const auto& mass = grid.fields().mass;
    const double h = grid.smoothing_length();
    std::array<double, 3> a{};
    grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
        if (!contributes(d2, cutoff_squared) || !prepare(j))
            return;
        const double gradient = m4_negative_radial_derivative_over_q_inside_support_3d(Kokkos::sqrt(d2) / h);
        const double pair =
            (force_coefficient(i) + force_coefficient(j)) * (gradient * inverse_h5) * mass(j);
        for (int axis = 0; axis < 3; ++axis)
            a[axis] += pair * (double(position(i, axis)) - position(j, axis));
    });
    return a;
}

template <bool from_step, bool check_density = !from_step, class WallAcceleration>
Status evaluate(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                const Values::const_type& density, const Values& pressure,
                const Acceleration3D& acceleration, const WallAcceleration& wall_acceleration,
                const std::array<double, 3>* gravity) {
    const auto& rows = grid.owned_slots();
    const auto count = rows.extent(0);
    if constexpr (!from_step) {
        if (auto s = eos.validate(); !s.ok())
            return s;
        if (density.extent(0) != count || pressure.extent(0) != count ||
            acceleration.extent(0) != count ||
            (count && (!density.data() || !pressure.data() || !acceleration.data())) ||
            overlaps(density, pressure) || overlaps(density, acceleration) ||
            overlaps(pressure, acceleration))
            return {StatusCode::invalid_argument,
                    "WCSPH requires three disjoint fields over the owned rows"};
    }
    if (!count)
        return {};
    const auto n = grid.fields().ids.extent(0);
    const auto& ghost = grid.fields().ghost;
    const double h = grid.smoothing_length();
    const double cutoff_squared = strict_m4_cutoff_squared_3d(grid.support_radius(), h);
    const double inverse_h5 = 1 / (h * h * h * h * h);
    try {
        Kokkos::View<double*, Kokkos::HostSpace> coefficient(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "sph.pressure_over_density_squared"),
            n);
        Kokkos::View<std::uint8_t*, Kokkos::HostSpace> needed;
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> fluid_row;
        const Policy owned(exec, 0, count);
        // Grid construction admits only counts representable by LocalIndex. The
        // count itself is distinct from every slot and every owned row, even at capacity.
        const auto missing = static_cast<LocalIndex>(n);
        if (n != count) {
            if (gravity)
                fluid_row = decltype(fluid_row)(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.fluid_row"), n);
            if constexpr (from_step) {
                Kokkos::deep_copy(exec, fluid_row, missing);
            } else {
                // Standalone wall acceleration can alias the acceleration output.
                // Finish reading all contributing wall inputs before force writes.
                needed = decltype(needed)(Kokkos::view_alloc(exec, "sph.pressure.needed_ghost"), n);
                Kokkos::parallel_for("sph.pressure.mark_ghosts", owned, [&](std::size_t row) {
                    const auto i = rows(row);
                    if (gravity)
                        fluid_row(i) = static_cast<LocalIndex>(row);
                    grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
                        if (ghost(j) && contributes(d2, cutoff_squared))
                            needed(j) = 1;
                    });
                });
            }
        }
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.pressure.owned_eos", owned,
            [&](std::size_t row, std::size_t& errors) {
                const auto rho = density(row);
                if constexpr (check_density) {
                    if (!(rho > 0)) {
                        ++errors;
                        return;
                    }
                }
                const auto slot = rows(row);
                if constexpr (from_step) {
                    if (n != count)
                        fluid_row(slot) = static_cast<LocalIndex>(row);
                }
                if (!prepare_source(eos, rho, pressure(row), coefficient(slot)))
                    ++errors;
            },
            invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "invalid owned WCSPH density or pressure exceeds real range"};
        if (!from_step && n != count) {
            const NeighborView3D::Indices row_of_slot = fluid_row;
            const Values::const_type owned_pressure = pressure;
            Kokkos::parallel_reduce(
                "sph.pressure.ghost_eos", Policy(exec, 0, n),
                [&](std::size_t slot, std::size_t& errors) {
                    if (!needed(slot))
                        return;
                    const auto j = static_cast<LocalIndex>(slot);
                    const auto source = gravity
                        ? wall_coefficient(
                              grid, j, row_of_slot, density, owned_pressure,
                              wall_acceleration(j), *gravity, eos)
                        : ghost_source_coefficient(grid, j, eos);
                    if (source)
                        coefficient(j) = *source;
                    else
                        ++errors;
                },
                invalid);
            if (invalid)
                return {StatusCode::numerical_failure,
                        gravity ? "invalid Adami wall pressure or inverse Tait density"
                              : "invalid reconstructed ghost WCSPH density or pressure "
                                "exceeds real range"};
        }
        const decltype(coefficient)::const_type force_coefficient = coefficient;
        const auto evaluate_forces = [&](const auto& prepare) {
            Kokkos::parallel_reduce(
                "sph.pressure.acceleration", owned,
                [&](std::size_t row, std::size_t& errors) {
                    const auto a = acceleration_at_slot(grid, rows(row), force_coefficient,
                                                        cutoff_squared, inverse_h5, prepare);
                    for (int axis = 0; axis < 3; ++axis) {
                        if (fits_real(a[axis]))
                            acceleration(row, axis) = static_cast<real>(a[axis]);
                        else
                            ++errors;
                    }
                }, invalid);
        };
        if (from_step && n != count) {
            const NeighborView3D::Indices row_of_slot = fluid_row;
            const Values::const_type owned_pressure = pressure;
            bool invalid_wall = false;
            evaluate_forces([&](LocalIndex j) {
                if (fluid_row(j) != missing)
                    return true;
                if (invalid_wall)
                    return false;
                const auto source = wall_coefficient(grid, j, row_of_slot, density, owned_pressure,
                                                     wall_acceleration(j), *gravity, eos);
                if (!source) {
                    invalid_wall = true;
                    return false;
                }
                coefficient(j) = *source;
                // Only owned entries serve as row indices in Adami reconstruction.
                // For a wall, any nonsentinel value marks its coefficient ready
                // for subsequent neighbors in this Serial force evaluation.
                fluid_row(j) = 0;
                return true;
            });
            if (invalid_wall)
                return {StatusCode::numerical_failure, "invalid Adami wall pressure or inverse Tait density"};
        } else {
            evaluate_forces([](LocalIndex) { return true; });
        }
        if (invalid)
            return {StatusCode::numerical_failure,
                    "WCSPH pressure acceleration exceeds real range"};
        return {};
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "cannot allocate WCSPH pressure scratch"};
    } catch (const std::exception&) {
        return {StatusCode::execution_failure, "cannot evaluate WCSPH pressure acceleration"};
    }
}
} // namespace

std::size_t step_pressure_workspace_bytes_per_source(bool has_walls) {
    return sizeof(double) + (has_walls ? sizeof(LocalIndex) : 0);
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                            const Values::const_type& density, const Values& pressure,
                            const Acceleration3D& acceleration) {
    return evaluate<false>(exec, grid, eos, density, pressure, acceleration,
                           [](LocalIndex) { return std::array<double, 3>{}; }, nullptr);
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                            const Values::const_type& density, const Values& pressure,
                            const Acceleration3D& acceleration,
                            const WallAcceleration3D::const_type& wall_acceleration,
                            const std::array<double, 3>& gravity) {
    const auto rows = grid.owned_slots().extent(0), n = grid.fields().ids.extent(0);
    if (rows && n != rows && (wall_acceleration.extent(0) != n || !wall_acceleration.data()))
        return {StatusCode::invalid_argument, "Adami wall acceleration must follow source slots"};
    const auto at_slot = [&](LocalIndex j) {
        return std::array{wall_acceleration(j, 0), wall_acceleration(j, 1), wall_acceleration(j, 2)};
    };
    return evaluate<false>(exec, grid, eos, density, pressure, acceleration, at_slot, &gravity);
}

Status evaluate_step_pressure_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                                 const Values::const_type& density, const Values& pressure,
                                 const Acceleration3D& acceleration,
                                 const WallAccelerationAt3D& wall_acceleration,
                                 const std::array<double, 3>& gravity, bool initial_density) {
    if (initial_density)
        return evaluate<true, true>(exec, grid, eos, density, pressure, acceleration, wall_acceleration, &gravity);
    return evaluate<true>(exec, grid, eos, density, pressure, acceleration, wall_acceleration, &gravity);
}

} // namespace nexora::sph

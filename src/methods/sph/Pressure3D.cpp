#include "Pressure3D.hpp"

#include "AdamiWallPressure3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
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
    if (!Kokkos::isfinite(rho) || rho <= 0)
        return false;
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
    if (!fits_real(density) ||
        !prepare_source(eos, static_cast<real>(density), pressure, coefficient))
        return std::nullopt;
    return coefficient;
}

bool contributes(double distance_squared, double support_squared) {
    return distance_squared > 0 && distance_squared < support_squared;
}

std::array<double, 3>
acceleration_at_slot(const NeighborView3D& grid, LocalIndex i,
                     const Kokkos::View<const double*, Kokkos::HostSpace>& force_coefficient,
                     double support_squared, double inverse_h5) {
    const auto& position = grid.fields().position;
    const auto& mass = grid.fields().mass;
    const double h = grid.smoothing_length();
    std::array<double, 3> a{};
    grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
        if (!contributes(d2, support_squared))
            return;
        const double gradient = m4_negative_radial_derivative_over_q_3d(Kokkos::sqrt(d2) / h);
        const double pair =
            (force_coefficient(i) + force_coefficient(j)) * (gradient * inverse_h5) * mass(j);
        for (int axis = 0; axis < 3; ++axis)
            a[axis] += pair * (double(position(i, axis)) - position(j, axis));
    });
    return a;
}

struct WallSources {
    const WallAcceleration3D::const_type& acceleration;
    const std::array<double, 3>& gravity;
};

Status evaluate(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                const Values::const_type& density, const Values& pressure,
                const Acceleration3D& acceleration, const std::optional<WallSources>& walls) {
    if (auto s = eos.validate(); !s.ok())
        return s;
    if (walls && !std::all_of(walls->gravity.begin(), walls->gravity.end(),
                              [](double x) { return std::isfinite(x); }))
        return {StatusCode::invalid_argument, "wall gravity must be finite"};
    const auto& rows = grid.owned_slots();
    const auto count = rows.extent(0);
    if (density.extent(0) != count || pressure.extent(0) != count ||
        acceleration.extent(0) != count ||
        (count && (!density.data() || !pressure.data() || !acceleration.data())) ||
        overlaps(density, pressure) || overlaps(density, acceleration) ||
        overlaps(pressure, acceleration))
        return {StatusCode::invalid_argument,
                "WCSPH requires three disjoint fields over the owned rows"};
    if (!count)
        return {};
    const auto n = grid.fields().ids.extent(0);
    if (walls && n != count &&
        (walls->acceleration.extent(0) != n || !walls->acceleration.data()))
        return {StatusCode::invalid_argument, "Adami wall acceleration must follow source slots"};
    const auto& ghost = grid.fields().ghost;
    const double h = grid.smoothing_length();
    const double support = grid.support_radius();
    const double support_squared = support * support;
    const double inverse_h5 = 1 / (h * h * h * h * h);
    try {
        Kokkos::View<double*, Kokkos::HostSpace> coefficient(
            Kokkos::view_alloc(exec, Kokkos::WithoutInitializing,
                               "sph.pressure_over_density_squared"),
            n);
        Kokkos::View<std::uint8_t*, Kokkos::HostSpace> needed;
        Kokkos::View<LocalIndex*, Kokkos::HostSpace> fluid_row;
        // A Serial pass marks ghosts participating in a nonzero gradient. Distant
        // and coincident ghosts need no pressure or reconstruction of their own density.
        const Policy owned(exec, 0, count);
        if (n != count) {
            if (walls)
                fluid_row = decltype(fluid_row)(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.fluid_row"), n);
            needed = decltype(needed)(Kokkos::view_alloc(exec, "sph.pressure.needed_ghost"), n);
            Kokkos::parallel_for("sph.pressure.mark_ghosts", owned, [&](std::size_t row) {
                const auto i = rows(row);
                if (walls)
                    fluid_row(i) = static_cast<LocalIndex>(row);
                grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
                    if (ghost(j) && contributes(d2, support_squared))
                        needed(j) = 1;
                });
            });
        }
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.pressure.owned_eos", owned,
            [&](std::size_t row, std::size_t& errors) {
                if (!prepare_source(eos, density(row), pressure(row), coefficient(rows(row))))
                    ++errors;
            },
            invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "invalid owned WCSPH density or pressure exceeds real range"};
        if (n != count) {
            const NeighborView3D::Indices row_of_slot = fluid_row;
            const Values::const_type owned_pressure = pressure;
            Kokkos::parallel_reduce(
                "sph.pressure.ghost_eos", Policy(exec, 0, n),
                [&](std::size_t slot, std::size_t& errors) {
                    if (!needed(slot))
                        return;
                    const auto j = static_cast<LocalIndex>(slot);
                    const auto source = walls
                        ? wall_coefficient(
                              grid, j, row_of_slot, density, owned_pressure,
                              {walls->acceleration(j, 0), walls->acceleration(j, 1),
                               walls->acceleration(j, 2)},
                              walls->gravity, eos)
                        : ghost_source_coefficient(grid, j, eos);
                    if (source)
                        coefficient(j) = *source;
                    else
                        ++errors;
                },
                invalid);
            if (invalid)
                return {StatusCode::numerical_failure,
                        walls ? "invalid Adami wall pressure or inverse Tait density"
                              : "invalid reconstructed ghost WCSPH density or pressure "
                                "exceeds real range"};
        }
        const decltype(coefficient)::const_type force_coefficient = coefficient;
        Kokkos::parallel_reduce(
            "sph.pressure.acceleration", owned,
            [&](std::size_t row, std::size_t& errors) {
                const auto a =
                    acceleration_at_slot(grid, rows(row), force_coefficient, support_squared,
                                         inverse_h5);
                for (int axis = 0; axis < 3; axis++) {
                    if (fits_real(a[axis]))
                        acceleration(row, axis) = static_cast<real>(a[axis]);
                    else
                        ++errors;
                }
            },
            invalid);
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

std::size_t pressure_workspace_bytes_per_source(bool has_ghosts, bool adami_walls) {
    const auto per_ghost = sizeof(std::uint8_t) + (adami_walls ? sizeof(LocalIndex) : 0);
    return sizeof(double) + (has_ghosts ? per_ghost : 0);
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                            const Values::const_type& density, const Values& pressure,
                            const Acceleration3D& acceleration) {
    return evaluate(exec, grid, eos, density, pressure, acceleration, std::nullopt);
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const NeighborView3D& grid, const TaitEos& eos,
                            const Values::const_type& density, const Values& pressure,
                            const Acceleration3D& acceleration,
                            const WallAcceleration3D::const_type& wall_acceleration,
                            const std::array<double, 3>& gravity) {
    return evaluate(exec, grid, eos, density, pressure, acceleration,
                    WallSources{wall_acceleration, gravity});
}

} // namespace nexora::sph

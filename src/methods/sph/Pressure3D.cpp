#include "Pressure3D.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>

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
    return Kokkos::isfinite(value) && Kokkos::abs(value) <= std::numeric_limits<real>::max();
}

// log1p and expm1 retain small pressure perturbations near equilibrium.
double tait_pressure(double rho, const WcsphParameters& p, double bulk_modulus) {
    const double difference = rho - p.reference_density;
    const double log_ratio = Kokkos::abs(difference) <= .5 * p.reference_density
                                 ? Kokkos::log1p(difference / p.reference_density)
                                 : Kokkos::log(rho / p.reference_density);
    return bulk_modulus * Kokkos::expm1(p.gamma * log_ratio);
}

double inverse_tait(double pressure, const WcsphParameters& p, double bulk_modulus) {
    return p.reference_density * Kokkos::exp(Kokkos::log1p(pressure / bulk_modulus) / p.gamma);
}

double wall_pressure(const CellGrid& grid, LocalIndex wall, const std::array<double, 3>& gravity,
                     const Kokkos::View<LocalIndex*, Kokkos::HostSpace>& fluid_row,
                     const Values::const_type& density, const Values& pressure) {
    double numerator = 0;
    double weights = 0;
    const auto& fields = grid.fields();
    const double h = grid.smoothing_length();
    grid.for_each_neighbor_of_slot(wall, [&](LocalIndex f, double d2) {
        if (fields.ghost(f))
            return;
        // The common h^-3 normalization cancels in the weighted wall average.
        const double w = m4_weight_3d(Kokkos::sqrt(d2) / h);
        const auto row = fluid_row(f);
        double head = 0;
        for (int axis = 0; axis < 3; ++axis) {
            const double dx = double(fields.position(wall, axis)) - fields.position(f, axis);
            head += gravity[axis] * dx;
        }
        numerator += (pressure(row) + density(row) * head) * w;
        weights += w;
    });
    return numerator / weights;
}

bool contributes(double distance_squared, double support_squared) {
    return distance_squared > 0 && distance_squared < support_squared;
}

std::array<double, 3>
acceleration_at_slot(const CellGrid& grid, LocalIndex i,
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

} // namespace

std::size_t pressure_workspace_bytes_per_source(bool has_ghosts, bool fixed_walls) {
    const auto per_ghost = sizeof(std::uint8_t) + (fixed_walls ? sizeof(LocalIndex) : 0);
    return sizeof(double) + (has_ghosts ? per_ghost : 0);
}

Status WcsphParameters::validate() const {
    if (!std::isfinite(reference_density) || reference_density <= 0 ||
        !std::isfinite(sound_speed) || sound_speed <= 0 || !std::isfinite(gamma) || gamma <= 0)
        return {StatusCode::invalid_argument,
                "WCSPH reference density, sound speed and gamma must be finite and positive"};
    return {};
}

Status evaluate_pressure_3d(const Kokkos::Serial& exec, const CellGrid& grid,
                            const WcsphParameters& parameters, const Values::const_type& density,
                            const Values& pressure, const Acceleration3D& acceleration,
                            const std::array<double, 3>* wall_gravity) {
    if (auto s = parameters.validate(); !s.ok())
        return s;
    if (wall_gravity && !std::all_of(wall_gravity->begin(), wall_gravity->end(),
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
    const auto& ghost = grid.fields().ghost;
    const double h = grid.smoothing_length();
    const double support = grid.support_radius();
    const double support_squared = support * support;
    const double inverse_h5 = 1 / (h * h * h * h * h);
    const double bulk_modulus =
        parameters.reference_density * parameters.sound_speed * parameters.sound_speed /
        parameters.gamma;
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
            if (wall_gravity)
                fluid_row = decltype(fluid_row)(
                    Kokkos::view_alloc(exec, Kokkos::WithoutInitializing, "sph.wall.fluid_row"), n);
            needed = decltype(needed)(Kokkos::view_alloc(exec, "sph.pressure.needed_ghost"), n);
            Kokkos::parallel_for("sph.pressure.mark_ghosts", owned, [&](std::size_t row) {
                const auto i = rows(row);
                if (wall_gravity)
                    fluid_row(i) = static_cast<LocalIndex>(row);
                grid.for_each_neighbor_of_slot(i, [&](LocalIndex j, double d2) {
                    if (ghost(j) && contributes(d2, support_squared))
                        needed(j) = 1;
                });
            });
        }
        const auto prepare_source = [&](LocalIndex slot, double rho, real& p) {
            if (!Kokkos::isfinite(rho) || rho <= 0)
                return false;
            const double raw = tait_pressure(rho, parameters, bulk_modulus);
            if (!fits_real(raw))
                return false;
            p = static_cast<real>(raw);
            coefficient(slot) = double(p) / (rho * rho);
            return true;
        };
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.pressure.owned_eos", owned,
            [&](std::size_t row, std::size_t& errors) {
                if (!prepare_source(rows(row), density(row), pressure(row)))
                    ++errors;
            },
            invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "invalid owned WCSPH density or pressure exceeds real range"};
        if (n != count) {
            Kokkos::parallel_reduce(
                "sph.pressure.ghost_eos", Policy(exec, 0, n),
                [&](std::size_t slot, std::size_t& errors) {
                    if (!needed(slot))
                        return;
                    const auto j = static_cast<LocalIndex>(slot);
                    if (wall_gravity) {
                        const double raw =
                            wall_pressure(grid, j, *wall_gravity, fluid_row, density, pressure);
                        if (!fits_real(raw)) {
                            ++errors;
                            return;
                        }
                        const real p = static_cast<real>(raw);
                        const double rho = inverse_tait(p, parameters, bulk_modulus);
                        if (!fits_real(rho) || static_cast<real>(rho) <= 0) {
                            ++errors;
                            return;
                        }
                        const double wall_density = static_cast<real>(rho);
                        coefficient(j) = double(p) / (wall_density * wall_density);
                        return;
                    }
                    const double rho = density_at_slot_3d(grid, j);
                    real p{};
                    if (!fits_real(rho) || !prepare_source(j, static_cast<real>(rho), p))
                        ++errors;
                },
                invalid);
            if (invalid)
                return {StatusCode::numerical_failure,
                        wall_gravity ? "invalid Adami wall pressure or inverse Tait density"
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
} // namespace nexora::sph

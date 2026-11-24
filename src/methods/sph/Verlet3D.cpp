#include "Verlet3D.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

Status validate_length(double h) {
    if (!std::isfinite(h) || h <= 0)
        return {StatusCode::invalid_argument, "WCSPH smoothing length must be finite and positive"};
    return {};
}

Status validate_cfl(double cfl) {
    if (!(cfl > 0 && cfl <= 1))
        return {StatusCode::invalid_argument, "WCSPH CFL safety factor must be in (0,1]"};
    return {};
}

template <class A, class B> bool overlaps(const A& a, const B& b) {
    if (!a.span() || !b.span())
        return false;
    const auto before = std::less<const void*>{};
    return before(a.data(), b.data() + b.span()) && before(b.data(), a.data() + a.span());
}

template <class... Views> bool valid_fields(std::size_t n, const Views&... views) {
    return ((views.extent(0) == n && (!n || views.data() != nullptr)) && ...);
}

Status validate_gravity(const std::array<double, 3>& gravity) {
    for (const double component : gravity)
        if (!std::isfinite(component))
            return {StatusCode::invalid_argument, "WCSPH gravity must be finite"};
    return {};
}

Status check_step(double dt, const std::array<double, 3>& gravity) {
    if (!std::isfinite(dt) || dt <= 0)
        return {StatusCode::invalid_argument, "Verlet step must be finite and positive"};
    return validate_gravity(gravity);
}

bool fits_real(double value) {
    return Kokkos::abs(value) <= std::numeric_limits<real>::max();
}
} // namespace

Status WcsphVerletParameters::validate() const {
    if (auto status = eos.validate(); !status.ok())
        return status;
    if (auto status = validate_gravity(gravity); !status.ok())
        return status;
    if (auto status = validate_length(smoothing_length); !status.ok())
        return status;
    return validate_cfl(cfl);
}

Status kick_drift_3d(const Kokkos::Serial& exec, const CellGrid& accepted,
                     const Velocity3D::const_type& velocity,
                     const Acceleration3D::const_type& acceleration, double dt,
                     const Velocity3D& staged_position, const Velocity3D& half_velocity,
                     const std::array<double, 3>& gravity) {
    if (auto s = check_step(dt, gravity); !s.ok())
        return s;
    const auto& position = accepted.fields().position;
    const auto n = position.extent(0);
    const auto owned = accepted.owned_slots();
    if (!valid_fields(owned.extent(0), velocity, acceleration) ||
        !valid_fields(n, staged_position, half_velocity) ||
        overlaps(staged_position, half_velocity) || overlaps(staged_position, position) ||
        overlaps(staged_position, velocity) || overlaps(staged_position, acceleration) ||
        overlaps(half_velocity, position) || overlaps(half_velocity, velocity) ||
        overlaps(half_velocity, acceleration))
        return {StatusCode::invalid_argument,
                "Verlet drift fields have invalid extents, storage or output overlap"};
    std::size_t invalid{};
    Kokkos::parallel_reduce(
        "sph.verlet.kick_drift", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t i, std::size_t& errors) {
            const auto slot = owned(i);
            for (int axis = 0; axis < 3; ++axis) {
                const double kicked =
                    Kokkos::fma(.5 * dt, acceleration(i, axis) + gravity[axis],
                                double(velocity(i, axis)));
                if (!fits_real(kicked)) {
                    ++errors;
                    continue;
                }
                const real half = static_cast<real>(kicked);
                const double drifted = Kokkos::fma(dt, double(half), double(position(slot, axis)));
                half_velocity(slot, axis) = half;
                staged_position(slot, axis) = narrow_step_value(drifted);
            }
        },
        invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "Verlet half velocity exceeds real range"};
    return {};
}

Status finish_kick_3d(const Kokkos::Serial& exec, const CellGrid& refreshed,
                      const Velocity3D::const_type& half_velocity,
                      const Acceleration3D::const_type& refreshed_acceleration, double dt,
                      const Velocity3D& velocity, const std::array<double, 3>& gravity) {
    if (auto s = check_step(dt, gravity); !s.ok())
        return s;
    const auto& old_slot = refreshed.sorted_to_input();
    const auto owned = refreshed.owned_slots();
    const auto n = owned.extent(0);
    if (!valid_fields(old_slot.extent(0), half_velocity) ||
        !valid_fields(n, refreshed_acceleration, velocity) || overlaps(velocity, half_velocity) ||
        overlaps(velocity, refreshed_acceleration) ||
        overlaps(velocity, refreshed.fields().position))
        return {StatusCode::invalid_argument,
                "Verlet final kick fields have invalid extents, storage or output overlap"};
    Kokkos::parallel_for("sph.verlet.finish_kick", Policy(exec, 0, n), [&](std::size_t i) {
        for (int axis = 0; axis < 3; ++axis)
            velocity(i, axis) = narrow_step_value(Kokkos::fma(
                .5 * dt, refreshed_acceleration(i, axis) + gravity[axis],
                double(half_velocity(old_slot(owned(i)), axis))));
    });
    return {};
}

Result<double> cfl_step_3d(const Kokkos::Serial& exec, const Velocity3D::const_type& velocity,
                           double sound_speed, double cfl_coefficient, double smoothing_length) {
    if (auto s = validate_length(smoothing_length); !s.ok())
        return s;
    if (auto s = validate_cfl(cfl_coefficient); !s.ok())
        return s;
    if (!std::isfinite(sound_speed) || sound_speed <= 0)
        return Status{StatusCode::invalid_argument,
                      "WCSPH acoustic speed must be finite and positive"};
    const auto n = velocity.extent(0);
    if (n && !velocity.data())
        return Status{StatusCode::invalid_argument, "WCSPH accepted velocity has no storage"};
    double largest_square = 0;
    if (n) {
        Kokkos::parallel_reduce(
            "sph.verlet.maximum_speed", Policy(exec, 0, n),
            [&](std::size_t i, double& maximum) {
                const double x = velocity(i, 0);
                const double y = velocity(i, 1);
                const double z = velocity(i, 2);
                const double square = x * x + y * y + z * z;
                // A NaN row must also yield no usable acoustic step.
                maximum = std::max(maximum, Kokkos::isnan(square)
                                                ? std::numeric_limits<double>::infinity()
                                                : square);
            },
            Kokkos::Max<double>(largest_square));
    }
    const double vmax = Kokkos::sqrt(largest_square);
    const double step = cfl_coefficient * smoothing_length / (sound_speed + vmax);
    if (!std::isfinite(step) || step <= 0)
        return Status{StatusCode::numerical_failure, "invalid WCSPH CFL step"};
    return step;
}
} // namespace nexora::sph

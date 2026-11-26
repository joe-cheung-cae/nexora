#include "Verlet3D.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>

namespace nexora::sph {
namespace {
using Policy = Kokkos::RangePolicy<Kokkos::Serial, Kokkos::IndexType<std::size_t>>;

Status validate_length(real h) {
    if (!std::isfinite(h) || h <= 0)
        return {StatusCode::invalid_argument, "WCSPH smoothing length must be finite and positive"};
    return {};
}

Status validate_cfl(real cfl) {
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

Status validate_gravity(const std::array<real, 3>& gravity) {
    for (const real component : gravity)
        if (!std::isfinite(component))
            return {StatusCode::invalid_argument, "WCSPH gravity must be finite"};
    return {};
}

Status check_step(real dt, const std::array<real, 3>& gravity) {
    if (!std::isfinite(dt) || dt <= 0)
        return {StatusCode::invalid_argument, "Verlet step must be finite and positive"};
    return validate_gravity(gravity);
}

Status write_kick_drift(const Kokkos::Serial& exec, const CellGrid& accepted,
                        const Velocity3D::const_type& velocity,
                        const Acceleration3D::const_type& acceleration, real dt,
                        const Velocity3D& staged_position, const Velocity3D& half_velocity,
                        const std::array<real, 3>& gravity) {
    const auto& position = accepted.fields().position;
    const auto owned = accepted.owned_slots();
    const Policy rows(exec, 0, owned.extent(0));
    // Standalone callers receive validated outputs here.
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.verlet.kick_drift", rows,
        [&](std::size_t i, std::size_t& errors) {
            const auto slot = owned(i);
            for (int axis = 0; axis < 3; ++axis) {
                const real half = Kokkos::fma(real{.5} * dt, acceleration(i, axis) + gravity[axis],
                                                    velocity(i, axis));
                if (!Kokkos::isfinite(half)) {
                    ++errors;
                    continue;
                }
                const real drifted = Kokkos::fma(dt, half, position(slot, axis));
                half_velocity(slot, axis) = half;
                if (Kokkos::isfinite(drifted))
                    staged_position(slot, axis) = drifted;
                else
                    ++errors;
            }
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure,
                "Verlet half velocity or drifted position exceeds real range"};
    return {};
}

real speed_squared(const Velocity3D& velocity, std::size_t row) {
    const real x = velocity(row, 0), y = velocity(row, 1), z = velocity(row, 2);
    return x * x + y * y + z * z;
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
                     const Acceleration3D::const_type& acceleration, real dt,
                     const Velocity3D& staged_position, const Velocity3D& half_velocity,
                     const std::array<real, 3>& gravity) {
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
    return write_kick_drift(exec, accepted, velocity, acceleration, dt, staged_position,
                                    half_velocity, gravity);
}

void verlet_step::kick_drift(const Kokkos::Serial& exec, const CellGrid& accepted,
                             const Velocity3D::const_type& velocity,
                             const Acceleration3D::const_type& acceleration, real dt,
                             const Velocity3D& staged_position, const Velocity3D& half_velocity,
                             const std::array<real, 3>& gravity) {
    const auto& position = accepted.fields().position;
    const auto owned = accepted.owned_slots();
    // Grid admission rejects nonfinite drift before using either staged output.
    Kokkos::parallel_for("sph.verlet.kick_drift", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t row) {
            const auto slot = owned(row);
            for (int axis = 0; axis < 3; ++axis) {
                const real half = Kokkos::fma(real{.5} * dt, acceleration(row, axis) + gravity[axis],
                                              velocity(row, axis));
                half_velocity(slot, axis) = half;
                staged_position(slot, axis) = Kokkos::fma(dt, half, position(slot, axis));
            }
        });
}

Result<real> verlet_step::finish_kick(const Kokkos::Serial& exec, const CellGrid& refreshed,
                                     const Velocity3D::const_type& half_velocity,
                                     const Acceleration3D::const_type& acceleration, real dt,
                                     const Velocity3D& velocity, const std::array<real, 3>& gravity) {
    const auto& to_input = refreshed.sorted_to_input();
    const auto& owned = refreshed.owned_slots();
    if (!owned.extent(0))
        return real{0};
    std::size_t invalid{};
    real largest_square{};
    Kokkos::parallel_reduce("sph.verlet.finish_kick", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t row, real& maximum, std::size_t& errors) {
            const auto slot = to_input(owned(row));
            for (int axis = 0; axis < 3; ++axis) {
                velocity(row, axis) =
                    Kokkos::fma(real{.5} * dt, acceleration(row, axis) + gravity[axis],
                                half_velocity(slot, axis));
            }
            const real square = speed_squared(velocity, row);
            if (Kokkos::isfinite(square))
                maximum = std::max(maximum, square);
            else
                ++errors;
        }, Kokkos::Max<real>(largest_square), Kokkos::Sum<std::size_t>(invalid));
    if (invalid)
        return Status{StatusCode::numerical_failure, "Verlet final squared speed must be finite"};
    return largest_square;
}

Result<real> verlet_step::capture_velocity(const Kokkos::Serial& exec, const NeighborView3D::Indices& rows,
                                             const Velocity3D::const_type& input, const Velocity3D& output) {
    if (!rows.extent(0))
        return real{0};
    real largest_square{};
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.verlet.initial_velocity", Policy(exec, 0, rows.extent(0)),
        [&](std::size_t row, real& maximum, std::size_t& errors) {
            const auto slot = rows(row);
            for (int axis = 0; axis < 3; ++axis)
                output(row, axis) = input(slot, axis);
            const real square = speed_squared(output, row);
            if (Kokkos::isfinite(square))
                maximum = std::max(maximum, square);
            else
                ++errors;
        }, Kokkos::Max<real>(largest_square), Kokkos::Sum<std::size_t>(invalid));
    if (invalid)
        return Status{StatusCode::invalid_argument, "WCSPH initial squared speed must be finite"};
    return largest_square;
}

Result<double> verlet_step::acoustic_step(real largest_square, real sound_speed,
                                          real cfl_coefficient, real smoothing_length) {
    const real vmax = Kokkos::sqrt(largest_square);
    double step = double(cfl_coefficient) * smoothing_length / (double(sound_speed) + vmax);
    constexpr auto invalid_step = "invalid WCSPH CFL step";
    if (!std::isfinite(step) || step <= 0)
        return Status{StatusCode::numerical_failure, invalid_step};
    if constexpr (std::is_same_v<real, float>) {
        if (step < std::numeric_limits<real>::denorm_min())
            return Status{StatusCode::numerical_failure, invalid_step};
        step = std::min(step, static_cast<double>(std::numeric_limits<real>::max()));
        const auto rounded = static_cast<real>(step);
        // An admitted duration must remain within the CFL limit after narrowing.
        if (rounded > step)
            step = std::nextafter(rounded, real{0});
    }
    return step;
}
} // namespace nexora::sph

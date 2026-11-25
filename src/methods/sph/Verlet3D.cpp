#include "Verlet3D.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <type_traits>

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

template <bool FromStep>
Status write_kick_drift(const Kokkos::Serial& exec, const CellGrid& accepted,
                        const Velocity3D::const_type& velocity,
                        const Acceleration3D::const_type& acceleration, double dt,
                        const Velocity3D& staged_position, const Velocity3D& half_velocity,
                        const std::array<double, 3>& gravity) {
    const auto& position = accepted.fields().position;
    const auto owned = accepted.owned_slots();
    const Policy rows(exec, 0, owned.extent(0));
    if constexpr (FromStep && std::is_same_v<real, double>) {
        // The step builds its grid before consuming either staged output.
        // With finite accepted positions and positive finite dt, a nonfinite
        // half velocity necessarily makes its drift nonfinite. Grid admission
        // rejects that coordinate before conversion or candidate publication.
        Kokkos::parallel_for("sph.verlet.kick_drift", rows, [&](std::size_t i) {
            const auto slot = owned(i);
            for (int axis = 0; axis < 3; ++axis) {
                const double half = Kokkos::fma(.5 * dt, acceleration(i, axis) + gravity[axis],
                                                  velocity(i, axis));
                half_velocity(slot, axis) = half;
                staged_position(slot, axis) = Kokkos::fma(dt, half, position(slot, axis));
            }
        });
    } else {
        // Standalone callers receive validated outputs here. Float storage also
        // needs the double-range test before narrowing, even if casting a value
        // just above FLT_MAX could round back to a finite float.
        std::size_t invalid{};
        Kokkos::parallel_reduce("sph.verlet.kick_drift", rows,
            [&](std::size_t i, std::size_t& errors) {
                const auto slot = owned(i);
                for (int axis = 0; axis < 3; ++axis) {
                    const double kicked = Kokkos::fma(.5 * dt, acceleration(i, axis) + gravity[axis],
                                                        double(velocity(i, axis)));
                    if (!fits_real(kicked)) {
                        ++errors;
                        continue;
                    }
                    const real half = static_cast<real>(kicked);
                    const double drifted = Kokkos::fma(dt, double(half), double(position(slot, axis)));
                    half_velocity(slot, axis) = half;
                    if (fits_real(drifted))
                        staged_position(slot, axis) = static_cast<real>(drifted);
                    else
                        ++errors;
                }
            }, invalid);
        if (invalid)
            return {StatusCode::numerical_failure,
                    "Verlet half velocity or drifted position exceeds real range"};
    }
    return {};
}

template <class Velocity> double speed_squared(const Velocity& velocity, std::size_t row) {
    const double x = velocity(row, 0), y = velocity(row, 1), z = velocity(row, 2);
    return x * x + y * y + z * z;
}

// The standalone final kick publishes only velocity. A solver step also uses
// its maximum squared speed to admit the next step.
template <bool compute_speed>
Result<double> write_final_velocity(const Kokkos::Serial& exec, const CellGrid& grid,
                                    const Velocity3D::const_type& half,
                                    const Acceleration3D::const_type& acceleration, double dt,
                                    const Velocity3D& velocity, const std::array<double, 3>& gravity) {
    const auto& to_input = grid.sorted_to_input();
    const auto& owned = grid.owned_slots();
    if (!owned.extent(0))
        return 0.;
    const auto write_row = [&](std::size_t row) {
        const auto slot = to_input(owned(row));
        for (int axis = 0; axis < 3; ++axis) {
            const double value = Kokkos::fma(.5 * dt, acceleration(row, axis) + gravity[axis],
                                             double(half(slot, axis)));
            if (!fits_real(value))
                return false;
            velocity(row, axis) = static_cast<real>(value);
        }
        return true;
    };
    const Policy rows(exec, 0, owned.extent(0));
    std::size_t invalid{};
    double largest_square{};
    if constexpr (compute_speed) {
        Kokkos::parallel_reduce("sph.verlet.finish_kick", rows,
            [&](std::size_t row, double& maximum, std::size_t& errors) {
                if (write_row(row))
                    maximum = std::max(maximum, speed_squared(velocity, row));
                else
                    ++errors;
            }, Kokkos::Max<double>(largest_square), Kokkos::Sum<std::size_t>(invalid));
    } else {
        Kokkos::parallel_reduce("sph.verlet.finish_kick", rows,
            [&](std::size_t row, std::size_t& errors) { errors += !write_row(row); }, invalid);
    }
    if (invalid)
        return Status{StatusCode::numerical_failure, "Verlet final velocity exceeds real range"};
    return largest_square;
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
    return write_kick_drift<false>(exec, accepted, velocity, acceleration, dt, staged_position,
                                    half_velocity, gravity);
}

Status verlet_step::kick_drift(const Kokkos::Serial& exec, const CellGrid& accepted,
                               const Velocity3D::const_type& velocity,
                               const Acceleration3D::const_type& acceleration, double dt,
                               const Velocity3D& staged_position, const Velocity3D& half_velocity,
                               const std::array<double, 3>& gravity) {
    return write_kick_drift<true>(exec, accepted, velocity, acceleration, dt, staged_position,
                                   half_velocity, gravity);
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
    return write_final_velocity<false>(exec, refreshed, half_velocity, refreshed_acceleration, dt,
                                       velocity, gravity).status();
}

Result<double> verlet_step::finish_kick(const Kokkos::Serial& exec, const CellGrid& refreshed,
                                        const Velocity3D::const_type& half_velocity,
                                        const Acceleration3D::const_type& acceleration, double dt,
                                        const Velocity3D& velocity, const std::array<double, 3>& gravity) {
    return write_final_velocity<true>(exec, refreshed, half_velocity, acceleration, dt, velocity, gravity);
}

Result<double> verlet_step::capture_velocity(const Kokkos::Serial& exec, const NeighborView3D::Indices& rows,
                                             const Velocity3D::const_type& input, const Velocity3D& output) {
    if (!rows.extent(0))
        return 0.;
    double largest_square{};
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.verlet.initial_velocity", Policy(exec, 0, rows.extent(0)),
        [&](std::size_t row, double& maximum, std::size_t& errors) {
            const auto slot = rows(row);
            for (int axis = 0; axis < 3; ++axis)
                output(row, axis) = input(slot, axis);
            const double square = speed_squared(output, row);
            if (!Kokkos::isfinite(square)) {
                // A finite double velocity can overflow when squared. In that
                // case acoustic_step reports the unusable CFL bound.
                for (int axis = 0; axis < 3; ++axis)
                    if (!Kokkos::isfinite(output(row, axis))) {
                        ++errors;
                        return;
                    }
            }
            maximum = std::max(maximum, square);
        }, Kokkos::Max<double>(largest_square), Kokkos::Sum<std::size_t>(invalid));
    if (invalid)
        return Status{StatusCode::invalid_argument, "WCSPH initial velocity must be finite"};
    return largest_square;
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
        std::size_t invalid{};
        Kokkos::parallel_reduce(
            "sph.verlet.maximum_speed", Policy(exec, 0, n),
            [&](std::size_t i, double& maximum, std::size_t& errors) {
                const double square = speed_squared(velocity, i);
                if (Kokkos::isfinite(square))
                    maximum = std::max(maximum, square);
                else
                    ++errors;
            }, Kokkos::Max<double>(largest_square), Kokkos::Sum<std::size_t>(invalid));
        if (invalid)
            return Status{StatusCode::numerical_failure, "invalid WCSPH CFL step"};
    }
    return verlet_step::acoustic_step(largest_square, sound_speed, cfl_coefficient, smoothing_length);
}

Result<double> verlet_step::acoustic_step(double largest_square, double sound_speed,
                                          double cfl_coefficient, double smoothing_length) {
    const double vmax = Kokkos::sqrt(largest_square);
    const double step = cfl_coefficient * smoothing_length / (sound_speed + vmax);
    if (!std::isfinite(step) || step <= 0)
        return Status{StatusCode::numerical_failure, "invalid WCSPH CFL step"};
    return step;
}
} // namespace nexora::sph

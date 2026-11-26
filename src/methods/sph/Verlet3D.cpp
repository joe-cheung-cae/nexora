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

std::size_t mapped_row(const NeighborView3D::Indices& rows, std::size_t row) {
    return rows.extent(0) ? rows(row) : row;
}

real slope_increment(real dt, real previous_dt, real rate_change) {
    // The slope coefficient alone can overflow or underflow before multiplication by a rate.
    if (!std::isfinite(rate_change))
        return rate_change;
    int dt_exponent, previous_exponent, rate_exponent;
    const real fraction = std::frexp(dt, &dt_exponent);
    const real previous = std::frexp(previous_dt, &previous_exponent);
    const real change = std::frexp(rate_change, &rate_exponent);
    return std::ldexp(fraction * fraction * change / previous,
                      2 * dt_exponent - previous_exponent + rate_exponent - 1);
}

struct IntegrateQuantity3D {
    using TimeScheme = central_difference_step::TimeScheme3D;

    explicit IntegrateQuantity3D(central_difference_step::TimeWeights3D time)
        : scheme(time.scheme), dt(time.dt), previous_dt(time.previous_dt),
          centered_duration(scheme == TimeScheme::centered ? time.dt + time.dt : real{0}),
          slope_weight(scheme == TimeScheme::adams_bashforth
                           ? dt * (real{.5} * (dt / time.previous_dt)) : real{0}) {}

    template <class PreviousValue, class PreviousRate>
    real operator()(real value, real rate, const PreviousValue& previous_value,
                    const PreviousRate& previous_rate, real constant_rate = 0) const {
        const real total_rate = rate + constant_rate;
        if (scheme == TimeScheme::centered) {
            if (Kokkos::isfinite(centered_duration))
                return Kokkos::fma(centered_duration, total_rate, previous_value());
            return Kokkos::fma(dt, total_rate, Kokkos::fma(dt, total_rate, previous_value()));
        }
        const real euler = Kokkos::fma(dt, total_rate, value);
        if (scheme == TimeScheme::adams_bashforth) {
            const real rate_change = rate - previous_rate();
            // A constant derivative needs no extrapolation, even after extreme step growth.
            if (rate_change != 0) {
                if (Kokkos::isfinite(slope_weight) && slope_weight >= std::numeric_limits<real>::min())
                    return Kokkos::fma(slope_weight, rate_change, euler);
                return euler + slope_increment(dt, previous_dt, rate_change);
            }
        }
        return euler;
    }

    TimeScheme scheme;
    real dt;
    real previous_dt;
    real centered_duration;
    real slope_weight;
};

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

void central_difference_step::drift_positions(const Kokkos::Serial& exec, const CellGrid& accepted,
    const Velocity3D::const_type& velocity, const Acceleration3D::const_type& pressure_acceleration,
    real dt, const Velocity3D& staged_position, const std::array<real, 3>& gravity) {
    const auto& position = accepted.fields().position;
    const auto owned = accepted.owned_slots();
    Kokkos::parallel_for("sph.central_difference.drift_positions", Policy(exec, 0, owned.extent(0)),
        [&](std::size_t row) {
            const auto slot = owned(row);
            for (int axis = 0; axis < 3; ++axis) {
                const real acceleration = pressure_acceleration(row, axis) + gravity[axis];
                const real midpoint_velocity = Kokkos::fma(real{.5} * dt, acceleration, velocity(row, axis));
                staged_position(slot, axis) = Kokkos::fma(dt, midpoint_velocity, position(slot, axis));
            }
        });
}

Result<real> central_difference_step::advance_velocity(const Kokkos::Serial& exec,
    const NeighborView3D::Indices& next_to_current, const NeighborView3D::Indices& current_to_previous,
    const Velocity3D::const_type& velocity, const Acceleration3D::const_type& pressure_acceleration,
    const Velocity3D::const_type& previous_velocity,
    const Acceleration3D::const_type& previous_pressure_acceleration,
    TimeWeights3D time, const Velocity3D& next_velocity, const std::array<real, 3>& gravity) {
    if (!next_velocity.extent(0))
        return real{0};
    const IntegrateQuantity3D integrate(time);
    real largest_square{};
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.central_difference.advance_velocity", Policy(exec, 0, next_velocity.extent(0)),
        [&](std::size_t next, real& maximum, std::size_t& errors) {
            const auto current = mapped_row(next_to_current, next);
            const auto previous = mapped_row(current_to_previous, current);
            for (int axis = 0; axis < 3; ++axis) {
                next_velocity(next, axis) = integrate(velocity(current, axis), pressure_acceleration(current, axis),
                    [&] { return previous_velocity(previous, axis); },
                    [&] { return previous_pressure_acceleration(previous, axis); }, gravity[axis]);
            }
            const real square = speed_squared(next_velocity, next);
            if (Kokkos::isfinite(square))
                maximum = std::max(maximum, square);
            else
                ++errors;
        }, Kokkos::Max<real>(largest_square), Kokkos::Sum<std::size_t>(invalid));
    if (invalid)
        return Status{StatusCode::numerical_failure, "WCSPH final squared speed must be finite"};
    return largest_square;
}

Status central_difference_step::advance_density(const Kokkos::Serial& exec,
    const NeighborView3D::Indices& next_to_current, const NeighborView3D::Indices& current_to_previous,
    const Values::const_type& density, const Values::const_type& density_rate,
    const Values::const_type& previous_density, const Values::const_type& previous_density_rate,
    TimeWeights3D time, const Values& next_density) {
    const IntegrateQuantity3D integrate(time);
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.central_difference.advance_density", Policy(exec, 0, next_density.extent(0)),
        [&](std::size_t next, std::size_t& errors) {
            const auto current = mapped_row(next_to_current, next);
            const auto previous = mapped_row(current_to_previous, current);
            const real value = integrate(density(current), density_rate(current),
                [&] { return previous_density(previous); }, [&] { return previous_density_rate(previous); });
            errors += !store_positive_density(value, next_density(next));
        }, invalid);
    if (invalid)
        return {StatusCode::numerical_failure, "advanced owned WCSPH density is outside positive real range"};
    return {};
}

Result<real> central_difference_step::capture_velocity(const Kokkos::Serial& exec,
    const NeighborView3D::Indices& rows, const Velocity3D::const_type& input, const Velocity3D& output) {
    if (!rows.extent(0))
        return real{0};
    real largest_square{};
    std::size_t invalid{};
    Kokkos::parallel_reduce("sph.central_difference.initial_velocity", Policy(exec, 0, rows.extent(0)),
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

Result<double> central_difference_step::acoustic_step(real largest_square, real sound_speed,
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

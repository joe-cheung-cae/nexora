#include "PrescribedMovingWall3D.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace nexora::sph::SphBoundary {
namespace {
using Vector = std::array<double, 3>;
Vector cross(const Vector& a, const Vector& b) {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}
std::optional<std::array<real, 3>> store_vector(const Vector& value) {
    std::array<real, 3> stored;
    for (int k = 0; k < 3; ++k) {
        if (!(std::abs(value[k]) <= std::numeric_limits<real>::max()))
            return std::nullopt;
        stored[k] = static_cast<real>(value[k]);
    }
    return stored;
}
Vector arm_at(const std::array<real, 3>& reference, const Vector& center) {
    return {double(reference[0]) - center[0], double(reference[1]) - center[1],
            double(reference[2]) - center[2]};
}
double periodic_phase(WallTrajectoryTime3D time, double period) {
    const auto initial = std::fmod(time.initial_phase, period);
    const auto elapsed = std::fmod(time.elapsed, period);
    double folded;
    if (initial < 0) {
        // A small negative initial phase may cancel directly against elapsed.
        // Adding a large period to it before that cancellation would erase it.
        folded = elapsed + initial;
        if (folded < 0) {
            folded = -initial >= .5 * period ? (period + initial) + elapsed
                                             : period + folded;
        }
    } else {
        const auto hi = std::max(initial, elapsed), lo = std::min(initial, elapsed);
        // During a wrap, hi >= period/2, so period-hi is exact and cannot overflow.
        const double gap = period - hi;
        folded = lo >= gap ? lo - gap : hi + lo;
    }
    return folded < period ? folded : 0.;
}

double difference_residual(double high, double low, double difference) {
    // FastTwoSum needs the larger-magnitude addend first. With a finite
    // rounded sum, this also keeps the intermediate subtraction finite.
    double a = high;
    double b = -low;
    if (std::abs(a) < std::abs(b))
        std::swap(a, b);
    return b - (difference - a);
}

double inward_difference(double high, double low) {
    const double difference = high - low;
    if (!std::isfinite(difference))
        return difference;
    // Round inward when the nearest representable difference exceeds the exact width.
    const double residual = difference_residual(high, low, difference);
    return residual < 0 ? std::nextafter(difference, 0.) : difference;
}
// Application-provided trajectories are called outside Kokkos kernels. Convert
// exceptions here so direct method calls and participant steps share one result contract.
template <class F, class... Args> auto call_trajectory(const F& f, Args... args)
    -> decltype(f(args...)) {
    try {
        return f(args...);
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "WCSPH trajectory callback allocation failed"};
    } catch (...) {
        return Status{StatusCode::execution_failure, "WCSPH trajectory callback failed"};
    }
}
} // namespace

PrescribedMovingWall3D::PrescribedMovingWall3D(ParticleView particles, PrescribedWallTrajectory3D trajectory)
    : SolidWallBoundary3D(std::move(particles)), trajectory_(std::move(trajectory)),
      last_elapsed_(inward_difference(trajectory_.last_phase, trajectory_.initial_phase)) {}

bool PrescribedMovingWall3D::in_domain(WallTrajectoryTime3D time) const {
    // Anchor the domain at initialization. A large initial phase can round a
    // still-interior sample up to last_phase if the two clock components are added first.
    const auto origin = time.initial_phase;
    return origin == trajectory_.initial_phase &&
           origin >= trajectory_.first_phase && origin <= trajectory_.last_phase &&
           time.elapsed >= 0 && std::isfinite(origin + time.elapsed) &&
           time.elapsed <= last_elapsed_;
}

Status PrescribedMovingWall3D::validate() const {
    return initial_step_limit().status();
}

Result<double> PrescribedMovingWall3D::initial_step_limit() const {
    if (!trajectory_.motion ||
        !(trajectory_.period >= 0 && std::isfinite(trajectory_.period)) ||
        !(trajectory_.time_resolution >= 0 && std::isfinite(trajectory_.time_resolution)))
        return Status{StatusCode::invalid_argument, "invalid WCSPH moving wall trajectory"};
    return maximum_sample_step(initial_time());
}

Result<WallTrajectoryTime3D> PrescribedMovingWall3D::advance_time(
    WallTrajectoryTime3D start, double dt, double certified_limit) const {
    if (!in_domain(start))
        return Status{StatusCode::numerical_failure, "WCSPH wall time span exceeds trajectory certification"};
    return advance_endpoint(start, dt, certified_limit);
}

Result<WallTrajectoryTime3D> PrescribedMovingWall3D::advance_endpoint(
    WallTrajectoryTime3D start, double dt, double limit) const {
    const WallTrajectoryTime3D end{start.initial_phase, start.elapsed + dt};
    const double delta = end.elapsed - start.elapsed;
    if (!(delta > 0 && delta <= limit && end.elapsed <= last_elapsed_) ||
        !std::isfinite(end.initial_phase + end.elapsed) ||
        !(std::abs(delta - dt) <= trajectory_.time_resolution))
        return Status{StatusCode::numerical_failure, "WCSPH wall time span exceeds trajectory certification"};
    return end;
}

Result<std::pair<WallTrajectoryTime3D, double>> PrescribedMovingWall3D::advance_certified_time(
    WallTrajectoryTime3D start, double dt, double limit) const {
    const auto end = advance_endpoint(start, dt, limit);
    if (!end.ok())
        return end.status();
    const auto next_limit = step_limit_in_domain(end.value());
    if (!next_limit.ok())
        return next_limit.status();
    return std::pair{end.value(), next_limit.value()};
}

Result<double> PrescribedMovingWall3D::maximum_sample_step(WallTrajectoryTime3D time) const {
    if (!in_domain(time))
        return Status{StatusCode::invalid_argument, "WCSPH wall cannot certify this trajectory time"};
    if (!(trajectory_.sample_step_limit > 0 && std::isfinite(trajectory_.sample_step_limit)))
        return Status{StatusCode::invalid_argument,
                      "WCSPH wall requires a finite positive sampling limit valid throughout its trajectory"};
    return step_limit_in_domain(time);
}

Result<double> PrescribedMovingWall3D::step_limit_in_domain(WallTrajectoryTime3D time) const {
    // An inward-rounded last elapsed value can still be strictly inside the domain.
    const double width = trajectory_.last_phase - time.initial_phase;
    if (time.elapsed == width &&
        difference_residual(trajectory_.last_phase, time.initial_phase, width) == 0)
        return 0.;
    const double remaining = inward_difference(last_elapsed_, time.elapsed);
    const double safe = std::min(trajectory_.sample_step_limit, remaining);
    const double next_elapsed = std::nextafter(time.elapsed, std::numeric_limits<double>::infinity());
    if (!(next_elapsed - time.elapsed <= safe))
        return Status{StatusCode::numerical_failure, "WCSPH wall has no positive remaining trajectory interval"};
    return safe;
}

Result<SampledWallMotion3D>
PrescribedMovingWall3D::motion_at_certified_time(WallTrajectoryTime3D time, WallMotionUse3D use) const {
    const double phase = trajectory_.period > 0 ? periodic_phase(time, trajectory_.period)
                                                : time.initial_phase + time.elapsed;
    auto sampled = call_trajectory(trajectory_.motion, time, phase);
    if (!sampled.ok())
        return sampled.status();
    return SampledWallMotion3D{std::move(sampled).value(), use};
}

SampledWallMotion3D::SampledWallMotion3D(RigidWallMotion3D sampled, WallMotionUse3D use)
    : motion(std::move(sampled)) {
    auto& m = motion;
    if ((use == WallMotionUse3D::position && m.angle == 0) ||
        (use == WallMotionUse3D::velocity && m.angular_speed == 0) ||
        (m.angle == 0 && m.angular_speed == 0 && m.angular_acceleration == 0))
        return;

    // Rescaling preserves finite subnormal and near-maximum directions.
    // An invalid axis propagates through any rotation that actually uses it.
    const double scale = std::max({std::abs(m.axis[0]), std::abs(m.axis[1]), std::abs(m.axis[2])});
    for (auto& component : m.axis)
        component /= scale;
    const double norm = std::hypot(m.axis[0], m.axis[1], m.axis[2]);
    for (auto& component : m.axis)
        component /= norm;
    if (m.angle != 0) {
        sine = std::sin(m.angle);
        const double half_sine = std::sin(.5 * m.angle);
        one_minus_cosine = 2 * half_sine * half_sine;
    }
}

Vector SampledWallMotion3D::displacement(const Vector& arm) const {
    const auto tangent = cross(motion.axis, arm);
    const auto normal = cross(motion.axis, tangent);
    Vector result;
    for (int k = 0; k < 3; ++k)
        result[k] = sine * tangent[k] + one_minus_cosine * normal[k];
    return result;
}

Vector SampledWallMotion3D::direction(const std::array<real, 3>& reference) const {
    auto rotated = arm_at(reference, motion.center);
    if (motion.angle != 0) {
        const auto offset = displacement(rotated);
        for (int k = 0; k < 3; ++k)
            rotated[k] += offset[k];
    }
    return cross(motion.axis, rotated);
}

Vector SampledWallMotion3D::position(const std::array<real, 3>& reference) const {
    Vector position;
    for (int k = 0; k < 3; ++k)
        position[k] = double(reference[k]) + motion.translation[k];
    if (motion.angle != 0) {
        const auto offset = displacement(arm_at(reference, motion.center));
        for (int k = 0; k < 3; ++k)
            position[k] += offset[k];
    }
    return position;
}

std::optional<std::array<real, 3>>
SampledWallMotion3D::checked_position(const std::array<real, 3>& reference) const {
    return store_vector(position(reference));
}

std::optional<std::array<real, 3>> SampledWallMotion3D::velocity(const std::array<real, 3>& reference) {
    if (motion.angular_speed == 0) {
        // Every particle has the same translational velocity at this stage.
        // Cache even a failed conversion so a later Serial row cannot repeat it.
        if (!checked_translation_velocity_) {
            translation_velocity_ = store_vector(motion.velocity);
            checked_translation_velocity_ = true;
        }
        return translation_velocity_;
    }
    const auto tangent = direction(reference);
    auto velocity = motion.velocity;
    for (int k = 0; k < 3; ++k)
        velocity[k] += motion.angular_speed * tangent[k];
    return store_vector(velocity);
}

Vector SampledWallMotion3D::acceleration(const std::array<real, 3>& reference) const {
    auto acceleration = motion.acceleration;
    if (motion.angular_speed != 0 || motion.angular_acceleration != 0) {
        const auto tangent = direction(reference);
        Vector rotational_velocity;
        for (int k = 0; k < 3; ++k)
            rotational_velocity[k] = motion.angular_speed * tangent[k];
        const auto centripetal = cross(motion.axis, rotational_velocity);
        for (int k = 0; k < 3; ++k)
            acceleration[k] += motion.angular_acceleration * tangent[k] + motion.angular_speed * centripetal[k];
    }
    return acceleration;
}

Result<WallKinematics3D> PrescribedMovingWall3D::sample_reference(
    const std::array<real, 3>& reference, WallTrajectoryTime3D time) const {
    if (!in_domain(time) || !trajectory_.motion)
        return Status{StatusCode::invalid_argument, "WCSPH wall sample outside its trajectory"};
    auto motion = motion_at_certified_time(time);
    if (!motion.ok())
        return motion.status();
    auto& sample = motion.value();
    const auto position = sample.checked_position(reference), velocity = sample.velocity(reference);
    const auto acceleration = sample.acceleration(reference);
    if (!position || !velocity ||
        !std::all_of(acceleration.begin(), acceleration.end(), [](double a) { return std::isfinite(a); }))
        return Status{StatusCode::numerical_failure, "WCSPH wall sample exceeds field range"};
    return WallKinematics3D{*position, *velocity, acceleration};
}

} // namespace nexora::sph::SphBoundary

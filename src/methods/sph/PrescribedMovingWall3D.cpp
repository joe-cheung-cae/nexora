#include "PrescribedMovingWall3D.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace nexora::sph::SphBoundary {
namespace {
using Vector = std::array<real, 3>;
Vector cross(const Vector& a, const Vector& b) {
    return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}
bool finite_vector(const Vector& value) {
    return std::all_of(value.begin(), value.end(), [](real x) { return std::isfinite(x); });
}

Vector arm_at(const Vector& reference, const Vector& center) {
    return {reference[0] - center[0], reference[1] - center[1], reference[2] - center[2]};
}
// Convert exceptions from application callbacks into step failures.
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
      last_elapsed_(trajectory_.last_phase - trajectory_.initial_phase) {}

Result<double> PrescribedMovingWall3D::initial_step_limit() const {
    if (!trajectory_.motion ||
        !(trajectory_.period >= 0 && std::isfinite(trajectory_.period)) ||
        !(trajectory_.time_resolution >= 0 && std::isfinite(trajectory_.time_resolution)))
        return Status{StatusCode::invalid_argument, "invalid WCSPH moving wall trajectory"};
    const auto start = initial_time();
    if (!(trajectory_.first_phase <= start.initial_phase && start.initial_phase <= trajectory_.last_phase &&
          std::isfinite(start.initial_phase)))
        return Status{StatusCode::invalid_argument, "WCSPH initial wall phase outside trajectory"};
    if (!(trajectory_.sample_step_limit > 0 && std::isfinite(trajectory_.sample_step_limit)))
        return Status{StatusCode::invalid_argument,
                      "WCSPH wall requires a finite positive sampling limit valid throughout its trajectory"};
    return step_limit_in_domain(start);
}

Result<std::pair<WallTrajectoryTime3D, double>> PrescribedMovingWall3D::advance_certified_time(
    WallTrajectoryTime3D start, double dt, double limit) const {
    const WallTrajectoryTime3D end{start.initial_phase, start.elapsed + dt};
    const double delta = end.elapsed - start.elapsed;
    if (!(delta > 0 && delta <= limit && end.elapsed <= last_elapsed_) ||
        !std::isfinite(end.initial_phase + end.elapsed) ||
        !(std::abs(delta - dt) <= trajectory_.time_resolution))
        return Status{StatusCode::numerical_failure, "WCSPH wall time span exceeds trajectory certification"};
    return std::pair{end, step_limit_in_domain(end)};
}

double PrescribedMovingWall3D::step_limit_in_domain(WallTrajectoryTime3D time) const {
    return std::min(trajectory_.sample_step_limit, last_elapsed_ - time.elapsed);
}

Result<SampledWallMotion3D>
PrescribedMovingWall3D::motion_at_certified_time(WallTrajectoryTime3D time, WallMotionUse3D use) const {
    double phase = time.initial_phase + time.elapsed;
    if (trajectory_.period > 0) {
        phase = std::fmod(phase, trajectory_.period);
        if (phase < 0)
            phase += trajectory_.period;
    }
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
    const real scale = std::max({std::abs(m.axis[0]), std::abs(m.axis[1]), std::abs(m.axis[2])});
    for (auto& component : m.axis)
        component /= scale;
    const real norm = std::sqrt(m.axis[0] * m.axis[0] + m.axis[1] * m.axis[1] + m.axis[2] * m.axis[2]);
    for (auto& component : m.axis)
        component /= norm;
    if (m.angle != 0) {
        sine = std::sin(m.angle);
        const real half_sine = std::sin(real{0.5} * m.angle);
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
        position[k] = reference[k] + motion.translation[k];
    if (motion.angle != 0) {
        const auto offset = displacement(arm_at(reference, motion.center));
        for (int k = 0; k < 3; ++k)
            position[k] += offset[k];
    }
    return position;
}

std::optional<std::array<real, 3>> SampledWallMotion3D::velocity(const std::array<real, 3>& reference) {
    if (motion.angular_speed == 0) {
        // Every particle has the same translational velocity at this stage.
        // Cache even a failed check so a later Serial row cannot repeat it.
        if (!checked_translation_velocity_) {
            if (finite_vector(motion.velocity))
                translation_velocity_ = motion.velocity;
            checked_translation_velocity_ = true;
        }
        return translation_velocity_;
    }
    const auto tangent = direction(reference);
    auto velocity = motion.velocity;
    for (int k = 0; k < 3; ++k)
        velocity[k] += motion.angular_speed * tangent[k];
    if (!finite_vector(velocity))
        return std::nullopt;
    return velocity;
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

} // namespace nexora::sph::SphBoundary

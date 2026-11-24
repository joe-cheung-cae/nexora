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
bool finite(const Vector& v) {
    return std::all_of(v.begin(), v.end(), [](double x) { return std::isfinite(x); });
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

double inward_difference(double high, double low) {
    const double difference = high - low;
    if (!std::isfinite(difference))
        return difference;
    // TwoDiff recovers the residual sign. An upper bound rounded outwards by
    // one fraction of an ulp must not certify an out-of-domain endpoint.
    const double virtual_low = high - difference;
    const double residual = (high - (difference + virtual_low)) + (virtual_low - low);
    return residual < 0 ? std::nextafter(difference, 0.) : difference;
}
// Application-provided trajectories are called outside Kokkos kernels. Convert
// exceptions here so direct method calls and participant windows share one result contract.
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
    : SolidWallBoundary3D(std::move(particles)), trajectory_(std::move(trajectory)) {}

bool PrescribedMovingWall3D::in_domain(WallTrajectoryTime3D time) const {
    // Anchor the domain at initialization. A large initial phase can round a
    // still-interior sample up to last_phase if the two clock components are added first.
    const auto origin = time.initial_phase;
    return origin == trajectory_.initial_phase && std::isfinite(origin) &&
           origin >= trajectory_.first_phase && origin <= trajectory_.last_phase &&
           time.elapsed >= 0 && time.elapsed <= inward_difference(trajectory_.last_phase, origin) &&
           std::isfinite(origin + time.elapsed);
}

Status PrescribedMovingWall3D::validate() const {
    if (!trajectory_.motion || !trajectory_.maximum_sample_step ||
        !(trajectory_.period >= 0 && std::isfinite(trajectory_.period)) ||
        !(trajectory_.time_resolution >= 0 && std::isfinite(trajectory_.time_resolution)) ||
        !in_domain(initial_time()))
        return {StatusCode::invalid_argument, "invalid WCSPH moving wall trajectory"};
    return {};
}

Result<WallTrajectoryTime3D> PrescribedMovingWall3D::advance_time(
    WallTrajectoryTime3D start, double dt, double certified_limit) const {
    const WallTrajectoryTime3D end{start.initial_phase, start.elapsed + dt};
    const double delta = end.elapsed - start.elapsed;
    if (!in_domain(start) || !in_domain(end) || !(delta > 0 && delta <= certified_limit) ||
        !(std::abs(delta - dt) <= trajectory_.time_resolution))
        return Status{StatusCode::numerical_failure, "WCSPH wall time span exceeds trajectory certification"};
    return end;
}

Result<double> PrescribedMovingWall3D::maximum_sample_step(WallTrajectoryTime3D time) const {
    if (!in_domain(time) || !trajectory_.maximum_sample_step)
        return Status{StatusCode::invalid_argument, "WCSPH wall cannot certify this trajectory time"};
    auto certified = call_trajectory(trajectory_.maximum_sample_step, time);
    if (!certified.ok())
        return certified.status();
    const auto span = certified.value();
    if (!(span > 0 && std::isfinite(span)))
        return Status{StatusCode::numerical_failure, "WCSPH wall requires a finite positive next step"};
    const double end_elapsed = inward_difference(trajectory_.last_phase, time.initial_phase);
    const double remaining = inward_difference(end_elapsed, time.elapsed);
    const double safe = std::min(span, remaining);
    const double next_elapsed = std::nextafter(time.elapsed, std::numeric_limits<double>::infinity());
    if (!(next_elapsed - time.elapsed <= safe))
        return Status{StatusCode::numerical_failure, "WCSPH wall has no positive remaining trajectory interval"};
    return safe;
}

Result<RigidWallMotion3D> PrescribedMovingWall3D::motion_at(WallTrajectoryTime3D time) const {
    if (!in_domain(time) || !trajectory_.motion)
        return Status{StatusCode::invalid_argument, "WCSPH wall sample outside its trajectory"};
    const double phase = trajectory_.period > 0 ? periodic_phase(time, trajectory_.period)
                                                : time.initial_phase + time.elapsed;
    auto sampled = call_trajectory(trajectory_.motion, time, phase);
    if (!sampled.ok())
        return sampled.status();
    auto& m = sampled.value();
    const double axis_scale = std::max({std::abs(m.axis[0]), std::abs(m.axis[1]), std::abs(m.axis[2])});
    if (!finite(m.translation) || !finite(m.velocity) || !finite(m.acceleration) ||
        !finite(m.center) || !finite(m.axis) || axis_scale == 0 || !std::isfinite(m.angle) ||
        !std::isfinite(m.angular_speed) || !std::isfinite(m.angular_acceleration))
        return Status{StatusCode::numerical_failure, "nonfinite WCSPH rigid motion or invalid rotation axis"};
    // The axis is a direction. Rescale before normalizing so even finite subnormal
    // or near-maximum components represent the same rotation as a unit direction.
    for (auto& component : m.axis)
        component /= axis_scale;
    const double norm = std::hypot(m.axis[0], m.axis[1], m.axis[2]);
    for (auto& component : m.axis)
        component /= norm;
    return sampled;
}

Result<WallKinematics3D> PrescribedMovingWall3D::apply(const std::array<real, 3>& reference,
                                                        const RigidWallMotion3D& m) {
    Vector position{}, velocity = m.velocity, acceleration = m.acceleration;
    if (m.angle == 0 && m.angular_speed == 0 && m.angular_acceleration == 0) {
        for (int k = 0; k < 3; ++k) position[k] = double(reference[k]) + m.translation[k];
    } else {
        Vector arm{};
        for (int k = 0; k < 3; ++k) arm[k] = double(reference[k]) - m.center[k];
        const auto tangent = cross(m.axis, arm);
        const auto normal = cross(m.axis, tangent);
        const double sine = std::sin(m.angle), half_sine = std::sin(.5 * m.angle);
        const double one_minus_cosine = 2 * half_sine * half_sine;
        Vector rotated{};
        for (int k = 0; k < 3; ++k) {
            const double displacement = sine * tangent[k] + one_minus_cosine * normal[k];
            rotated[k] = arm[k] + displacement;
            position[k] = double(reference[k]) + m.translation[k] + displacement;
        }
        const auto direction = cross(m.axis, rotated);
        Vector rotation_velocity{};
        for (int k = 0; k < 3; ++k) {
            rotation_velocity[k] = m.angular_speed * direction[k];
            velocity[k] += rotation_velocity[k];
        }
        const auto centripetal = cross(m.axis, rotation_velocity);
        for (int k = 0; k < 3; ++k)
            acceleration[k] += m.angular_acceleration * direction[k] + m.angular_speed * centripetal[k];
    }
    WallKinematics3D sample;
    for (int k = 0; k < 3; ++k) {
        constexpr double max = std::numeric_limits<real>::max();
        if (!(std::abs(position[k]) <= max && std::abs(velocity[k]) <= max && std::isfinite(acceleration[k])))
            return Status{StatusCode::numerical_failure, "WCSPH wall sample exceeds field range"};
        sample.position[k] = static_cast<real>(position[k]);
        sample.velocity[k] = static_cast<real>(velocity[k]);
        sample.acceleration[k] = acceleration[k];
    }
    return sample;
}

Result<WallKinematics3D> PrescribedMovingWall3D::sample_reference(
    const std::array<real, 3>& reference, WallTrajectoryTime3D time) const {
    const auto motion = motion_at(time);
    if (!motion.ok())
        return motion.status();
    return apply(reference, motion.value());
}

} // namespace nexora::sph::SphBoundary

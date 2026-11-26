#include "PrescribedMotion.hpp"

#include <lepton/Operation.h>
#include <lepton/ParsedExpression.h>
#include <lepton/Parser.h>

#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>

namespace nexora::app {
namespace {
namespace wall = sph::SphBoundary;

void check_variables(const Lepton::ExpressionTreeNode& node) {
    const auto& op = node.getOperation();
    if (op.getId() == Lepton::Operation::VARIABLE && op.getName() != "t")
        throw std::invalid_argument("unknown motion variable: " + op.getName());
    for (const auto& child : node.getChildren())
        check_variables(child);
}

real wall_value(double value) {
    if (!(std::abs(value) <= std::numeric_limits<real>::max()))
        throw std::range_error("prescribed motion value outside finite wall precision");
    return static_cast<real>(value);
}

struct Sample {
    double elapsed{}, physical_time{};
    wall::RigidWallMotion3D motion{};
};

struct Component {
    Lepton::ParsedExpression value, first;
    std::optional<Lepton::ParsedExpression> second;
};
} // namespace

struct PrescribedMotion::Impl {
    MotionMode mode;
    std::array<Component, 4> components;
    Sample accepted;
    std::optional<Sample> trial;

    Sample evaluate(double elapsed, double physical_time, double dt) const {
        Sample sample = accepted;
        sample.elapsed = elapsed;
        sample.physical_time = physical_time;
        const std::map<std::string, double> variables{{"t", physical_time}};
        for (std::size_t i = 0; i < components.size(); ++i) {
            const auto& component = components[i];
            const real value = wall_value(component.value.evaluate(variables));
            const real first = wall_value(component.first.evaluate(variables));
            real position = value, velocity = first, acceleration;
            if (mode == MotionMode::pose) {
                acceleration = wall_value(component.second->evaluate(variables));
            } else {
                const auto& previous = accepted.motion;
                const real start = i == 3 ? previous.angle : previous.translation[i];
                const real rate = i == 3 ? previous.angular_speed : previous.velocity[i];
                position = wall_value(std::fma(static_cast<real>(dt), rate, start));
                velocity = value;
                acceleration = first;
            }
            if (i == 3) {
                sample.motion.angle = position;
                sample.motion.angular_speed = velocity;
                sample.motion.angular_acceleration = acceleration;
            } else {
                sample.motion.translation[i] = position;
                sample.motion.velocity[i] = velocity;
                sample.motion.acceleration[i] = acceleration;
            }
        }
        return sample;
    }
};

PrescribedMotion::PrescribedMotion(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PrescribedMotion::~PrescribedMotion() = default;

Result<std::unique_ptr<PrescribedMotion>> PrescribedMotion::create(const MotionConfig& config) {
    try {
        if (config.mode != MotionMode::pose && config.mode != MotionMode::velocity)
            throw std::invalid_argument("unknown prescribed motion mode");
        if (!std::isfinite(config.initial_time))
            throw std::invalid_argument("prescribed motion requires a finite initial time");
        if (config.mode == MotionMode::pose && config.initial_pose != std::array<double, 4>{})
            throw std::invalid_argument("initial_pose applies only to prescribed velocity");
        auto impl = std::make_unique<Impl>();
        impl->mode = config.mode;
        auto& frame = impl->accepted.motion;
        for (std::size_t i = 0; i < frame.axis.size(); ++i) {
            frame.translation[i] = wall_value(config.initial_pose[i]);
            frame.axis[i] = wall_value(config.axis[i]);
            frame.center[i] = wall_value(config.center[i]);
        }
        frame.angle = wall_value(config.initial_pose[3]);
        if (frame.axis == std::array<real, 3>{})
            throw std::invalid_argument("prescribed rotation requires a nonzero axis");
        for (std::size_t i = 0; i < impl->components.size(); ++i) {
            const auto& source = config.expressions[i];
            auto& component = impl->components[i];
            component.value = Lepton::Parser::parse(source);
            check_variables(component.value.getRootNode());
            component.first = component.value.differentiate("t").optimize();
            if (config.mode == MotionMode::pose)
                component.second = component.first.differentiate("t").optimize();
        }
        impl->accepted = impl->evaluate(0, config.initial_time, 0);
        return std::unique_ptr<PrescribedMotion>(new PrescribedMotion(std::move(impl)));
    } catch (const std::bad_alloc&) {
        return Status{StatusCode::resource_exhausted, "prescribed motion allocation failed"};
    } catch (const std::exception& error) {
        return Status{StatusCode::invalid_argument, error.what()};
    }
}

wall::PrescribedWallTrajectory3D PrescribedMotion::trajectory(double sample_step_limit) const {
    wall::PrescribedWallTrajectory3D result;
    result.sample_step_limit = sample_step_limit;
    result.motion = [this](wall::WallTrajectoryTime3D time, double) -> Result<wall::RigidWallMotion3D> {
        if (time.elapsed == impl_->accepted.elapsed)
            return impl_->accepted.motion;
        if (impl_->trial && time.elapsed == impl_->trial->elapsed)
            return impl_->trial->motion;
        return Status{StatusCode::invalid_state, "prescribed wall time has no prepared sample"};
    };
    return result;
}

Status PrescribedMotion::prepare(double dt) {
    discard();
    const double elapsed = impl_->accepted.elapsed + dt;
    const double time = impl_->accepted.physical_time + dt;
    if (!(dt > 0 && elapsed > impl_->accepted.elapsed && std::isfinite(elapsed) && std::isfinite(time) &&
          time > impl_->accepted.physical_time))
        return {StatusCode::invalid_argument, "prescribed wall step must advance finite physical time"};
    try {
        if (impl_->mode == MotionMode::velocity && !(wall_value(dt) > 0))
            return {StatusCode::numerical_failure, "prescribed velocity step rounds to zero in wall precision"};
        impl_->trial = impl_->evaluate(elapsed, time, dt);
        return {};
    } catch (const std::bad_alloc&) {
        return {StatusCode::resource_exhausted, "prescribed motion allocation failed"};
    } catch (const std::exception& error) {
        return {StatusCode::numerical_failure, error.what()};
    }
}

void PrescribedMotion::commit() noexcept {
    if (impl_->trial) {
        impl_->accepted = *impl_->trial;
        discard();
    }
}

void PrescribedMotion::discard() noexcept { impl_->trial.reset(); }

} // namespace nexora::app

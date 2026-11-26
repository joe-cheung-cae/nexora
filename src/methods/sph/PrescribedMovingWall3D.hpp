#pragma once
#include "SphBoundary.hpp"

#include <memory>
#include <type_traits>
#include <utility>
#include <limits>
#include <optional>

namespace nexora::sph::SphBoundary {

// Rigid translation and fixed-axis rotation about center, with physical time derivatives.
struct RigidWallMotion3D {
    std::array<real, 3> translation{}, velocity{}, acceleration{}, center{}, axis{1, 0, 0};
    real angle{}, angular_speed{}, angular_acceleration{};
};

struct RigidWallPose3D {
    std::array<real, 3> translation{}, center{}, axis{1, 0, 0};
    real angle{};
};

struct WallMotionRates3D {
    std::array<real, 3> velocity{}, acceleration{};
    real angular_speed{}, angular_acceleration{};
};

// A trajectory callback is sampled once per group. Geometry and time derivatives
// are evaluated separately at the particle that consumes each quantity.
enum class WallMotionUse3D { position, full };

class SampledWallMotion3D {
  public:
    SampledWallMotion3D() = default;
    SampledWallMotion3D(RigidWallMotion3D, WallMotionUse3D);
    std::array<real, 3> position(const std::array<real, 3>& reference) const;
    std::optional<std::array<real, 3>> velocity(const std::array<real, 3>& reference);
    std::array<real, 3> acceleration(const std::array<real, 3>& reference) const;

  private:
    friend class PrescribedMovingWall3D;
    RigidWallMotion3D motion;
    real sine{}, one_minus_cosine{};
    bool checked_translation_velocity_{};
    std::optional<std::array<real, 3>> translation_velocity_;
    std::array<real, 3> displacement(const std::array<real, 3>& arm) const;
    std::array<real, 3> direction(const std::array<real, 3>& reference) const;
};

// A trajectory owns a measurable copy of its motion callback. Capture parameters directly
// in the callable, or borrow externally owned parameters that outlive the trajectory.
// Plain value storage makes the size of the retained allocation known before admission.
template <class R, class... Args> class WallTrajectoryFunction {
    struct Target {
        virtual ~Target() = default;
        virtual R call(Args... args) const = 0;
        virtual std::size_t bytes() const = 0;
        virtual std::unique_ptr<Target> clone() const = 0;
    };
    template <class F> struct Stored final : Target {
        F function;
        explicit Stored(F f) : function(f) {}
        R call(Args... args) const override { return function(args...); }
        std::size_t bytes() const override { return sizeof(Stored); }
        std::unique_ptr<Target> clone() const override { return std::make_unique<Stored>(function); }
    };
    std::unique_ptr<Target> target_;

  public:
    WallTrajectoryFunction() = default;
    template <class F> requires(std::is_trivially_copyable_v<F> &&
                                 std::is_invocable_r_v<R, const F&, Args...>)
    WallTrajectoryFunction(F function) {
        if constexpr (std::is_pointer_v<F>) {
            if (!function)
                return;
        }
        target_ = std::make_unique<Stored<F>>(function);
    }
    WallTrajectoryFunction(const WallTrajectoryFunction& other)
        : target_(other.target_ ? other.target_->clone() : nullptr) {}
    WallTrajectoryFunction(WallTrajectoryFunction&&) noexcept = default;
    WallTrajectoryFunction& operator=(WallTrajectoryFunction other) noexcept {
        target_.swap(other.target_);
        return *this;
    }
    explicit operator bool() const { return bool(target_); }
    std::size_t retained_bytes() const { return target_ ? target_->bytes() : 0; }
    R operator()(Args... args) const { return target_->call(args...); }
};

/// SPH owns the pose. The owner supplies physical rates at each integer time level.
/// Translation and angle advance using the accepted linear velocity and angular speed.
/// Center and axis stay fixed in reference coordinates.
struct VelocityDrivenWall3D {
    RigidWallPose3D initial_pose;
    WallTrajectoryFunction<Result<WallMotionRates3D>, WallTrajectoryTime3D, double> rates;
};

struct PrescribedWallTrajectory3D {
    double initial_phase{}, period{};
    double first_phase = -std::numeric_limits<double>::infinity();
    double last_phase = std::numeric_limits<double>::infinity();
    double time_resolution{};
    // Only the motion phase is reduced modulo period. Motion also receives the original clock.
    /// Owner-supplied absolute pose and physical rates, sampled directly at the requested phase.
    WallTrajectoryFunction<Result<RigidWallMotion3D>, WallTrajectoryTime3D, double> motion;
    // Required finite positive physical bound valid throughout the supplied trajectory.
    double sample_step_limit{};
    // Select exactly one of absolute motion and velocity-driven pose.
    std::optional<VelocityDrivenWall3D> velocity_driven{};

    std::size_t retained_bytes() const {
        return motion.retained_bytes() + (velocity_driven ? velocity_driven->rates.retained_bytes() : 0);
    }
};
struct WallGroupInput3D {
    std::uint64_t id;
    FixedWallParticles3D particles;
    std::optional<PrescribedWallTrajectory3D> trajectory;
};

class PrescribedMovingWall3D final : public SolidWallBoundary3D {
  public:
    PrescribedMovingWall3D(ParticleView particles, PrescribedWallTrajectory3D trajectory);
    bool velocity_driven() const { return trajectory_.velocity_driven.has_value(); }
    WallTrajectoryTime3D initial_time() const { return {trajectory_.initial_phase, 0}; }
    Result<double> initial_step_limit() const;
    // The time and limit must come from this trajectory's initialized or
    // previously certified state. The stepper retains them as a pair.
    Result<std::pair<WallTrajectoryTime3D, double>> advance_certified_time(
        WallTrajectoryTime3D start, double dt, double limit) const;
    // After initialization, a velocity-driven wall takes the accepted sample from
    // this group and the admitted step duration that advances its pose.
    Result<SampledWallMotion3D> motion_at_certified_time(WallTrajectoryTime3D time, WallMotionUse3D use,
        const SampledWallMotion3D* accepted = nullptr, real dt = 0) const;

  private:
    PrescribedWallTrajectory3D trajectory_;
    double last_elapsed_;
    double step_limit_in_domain(WallTrajectoryTime3D time) const;
};

} // namespace nexora::sph::SphBoundary

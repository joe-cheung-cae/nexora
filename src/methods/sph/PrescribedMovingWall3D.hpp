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
    std::array<double, 3> translation{}, velocity{}, acceleration{}, center{}, axis{1, 0, 0};
    double angle{}, angular_speed{}, angular_acceleration{};
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

struct PrescribedWallTrajectory3D {
    double initial_phase{}, period{};
    double first_phase = -std::numeric_limits<double>::infinity();
    double last_phase = std::numeric_limits<double>::infinity();
    double time_resolution{};
    // Only the motion phase is reduced modulo period. Motion also receives the original clock.
    WallTrajectoryFunction<Result<RigidWallMotion3D>, WallTrajectoryTime3D, double> motion;
    // Required finite positive physical bound valid throughout the supplied trajectory.
    double sample_step_limit{};
};
struct WallGroupInput3D {
    std::uint64_t id;
    FixedWallParticles3D particles;
    std::optional<PrescribedWallTrajectory3D> trajectory;
};

class PrescribedMovingWall3D final : public SolidWallBoundary3D {
  public:
    PrescribedMovingWall3D(ParticleView particles, PrescribedWallTrajectory3D trajectory);
    Status validate() const;
    WallTrajectoryTime3D initial_time() const { return {trajectory_.initial_phase, 0}; }
    Result<WallTrajectoryTime3D> advance_time(WallTrajectoryTime3D start, double dt,
                                              double certified_limit) const;
    Result<double> maximum_sample_step(WallTrajectoryTime3D time) const;
    Result<RigidWallMotion3D> motion_at(WallTrajectoryTime3D time) const;
    // motion_at validates the motion and normalizes the axis once per group and stage.
    static Result<WallKinematics3D> apply(const std::array<real, 3>& reference,
                                          const RigidWallMotion3D& motion);

  private:
    PrescribedWallTrajectory3D trajectory_;
    bool in_domain(WallTrajectoryTime3D time) const;
    Result<WallKinematics3D> sample_reference(const std::array<real, 3>& reference,
                                              WallTrajectoryTime3D time) const override;
};

} // namespace nexora::sph::SphBoundary

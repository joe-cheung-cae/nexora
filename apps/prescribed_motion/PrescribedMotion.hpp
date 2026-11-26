#pragma once

#include <methods/sph/PrescribedMovingWall3D.hpp>

#include <array>
#include <memory>
#include <string>

namespace nexora::app {

enum class MotionMode { pose, velocity };

struct MotionConfig {
    MotionMode mode{MotionMode::pose};
    // Translation along x, y, z and angle about the fixed axis, or their rates.
    std::array<std::string, 4> expressions{"0", "0", "0", "0"};
    std::array<double, 4> initial_pose{}; // Velocity mode only.
    std::array<double, 3> center{}, axis{1, 0, 0};
    double initial_time{}; // Physical simulation time at trajectory elapsed == 0.
};

class PrescribedMotion {
  public:
    static Result<std::unique_ptr<PrescribedMotion>> create(const MotionConfig&);
    ~PrescribedMotion();

    // This owner must outlive every copy of the returned trajectory.
    sph::SphBoundary::PrescribedWallTrajectory3D trajectory(double sample_step_limit) const;
    // Stage before the SPH trial. Publish only after all participants accept it.
    // Advance the simulation clock and the group's elapsed time independently by dt.
    Status prepare(double dt);
    void commit() noexcept;
    void discard() noexcept;

  private:
    struct Impl;
    explicit PrescribedMotion(std::unique_ptr<Impl>);
    std::unique_ptr<Impl> impl_;
};

} // namespace nexora::app

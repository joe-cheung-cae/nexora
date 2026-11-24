#pragma once

#include "ParticleFields3D.hpp"
#include <nexora/core/Result.hpp>

#include <array>

namespace nexora::sph {

struct FixedWallParticles3D {
    decltype(ParticleFields3D::position) position;
    decltype(ParticleFields3D::mass) mass;
    decltype(ParticleFields3D::ids) ids;
};

namespace SphBoundary {

struct WallParticle3D {
    std::uint64_t id;
    std::array<real, 3> reference_position;
    real mass;
};

struct WallTrajectoryTime3D {
    double initial_phase{}, elapsed{};
};

struct WallKinematics3D {
    std::array<real, 3> position, velocity;
    std::array<double, 3> acceleration;
};

class SolidWallBoundary3D {
  public:
    using ParticleView = Kokkos::View<const WallParticle3D*, Kokkos::HostSpace>;

    virtual ~SolidWallBoundary3D() = default;
    const ParticleView& particles() const { return particles_; }
    const WallParticle3D* find(std::uint64_t id) const;
    Result<WallKinematics3D> sample(std::uint64_t id, WallTrajectoryTime3D time) const;

  protected:
    explicit SolidWallBoundary3D(ParticleView sorted_particles);
    virtual Result<WallKinematics3D> sample_reference(const std::array<real, 3>& position,
                                                      WallTrajectoryTime3D time) const = 0;

  private:
    ParticleView particles_;
};

class FixedWall3D final : public SolidWallBoundary3D {
  public:
    explicit FixedWall3D(ParticleView sorted_particles);
    static WallKinematics3D kinematics(const std::array<real, 3>& position) {
        return {position, {}, {}};
    }

  private:
    Result<WallKinematics3D> sample_reference(const std::array<real, 3>& position,
                                              WallTrajectoryTime3D time) const override;
};

// Capture independent, sorted records. The caller admits input count times sizeof(WallParticle3D).
Result<SolidWallBoundary3D::ParticleView> capture_wall_particles(const Kokkos::Serial&,
                                                                const FixedWallParticles3D&);

} // namespace SphBoundary
} // namespace nexora::sph

#pragma once

#include "ParticleFields3D.hpp"
#include <nexora/core/Result.hpp>

#include <array>

namespace nexora::sph {

class WcsphStep3D;

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

class SolidWallBoundary3D {
  public:
    using ParticleView = Kokkos::View<const WallParticle3D*, Kokkos::HostSpace>;

    const ParticleView& particles() const { return particles_; }
    const WallParticle3D* find(std::uint64_t id) const;

  protected:
    explicit SolidWallBoundary3D(ParticleView sorted_particles);

  private:
    friend class ::nexora::sph::WcsphStep3D;
    // Production assembly validates masses and the combined ID set at its
    // initial grid boundary. Moving reference positions enter trajectory
    // arithmetic earlier and must be finite before invoking user motion.
    static Result<ParticleView> capture_for_grid(const Kokkos::Serial&,
                                                 const FixedWallParticles3D&, bool moving);
    ParticleView particles_;
};

class FixedWall3D final : public SolidWallBoundary3D {
  public:
    explicit FixedWall3D(ParticleView sorted_particles);
};

} // namespace SphBoundary
} // namespace nexora::sph

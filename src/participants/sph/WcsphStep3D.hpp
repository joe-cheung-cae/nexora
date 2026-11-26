#pragma once

#include "Geometry3D.hpp"

#include <methods/sph/Continuity3D.hpp>
#include <methods/sph/PrescribedMovingWall3D.hpp>
#include <methods/sph/Verlet3D.hpp>
#include <nexora/core/Time.hpp>
#include <nexora/data/FieldBinding.hpp>

#include <functional>
#include <optional>
#include <variant>
#include <vector>

namespace nexora::sph {

struct WallPhase3D {
    std::uint64_t group_id{};
    SphBoundary::WallTrajectoryTime3D time;
    double maximum_step{};
};
using WallPhases3D = Kokkos::View<const WallPhase3D*, Kokkos::HostSpace>;

/// Owns the fluid read leases and storage reservations transferred to history on commit.
struct FluidTimeLevel3D {
    detail::ResourceBudget::Ticket density_storage, velocity_storage, acceleration_storage, rate_storage;
    std::optional<ReadField<Values::const_type>> density;
    std::optional<ReadField<Velocity3D::const_type>> velocity, acceleration;
};

/// One preceding accepted time level. Its row map follows stable IDs across grid reordering.
struct IntegrationHistory3D : FluidTimeLevel3D {
    DensityRate3D::const_type density_rate;
    detail::ResourceBudget::Ticket row_storage;
    Geometry3D::Indices previous_row; // Current owned row -> same ID at the preceding level; empty = identity.
    double step_duration{};
    unsigned centered_steps{};
};

struct WcsphState3D : FluidTimeLevel3D {
    DensityRate3D::const_type density_rate;
    detail::ResourceBudget::Ticket phase_storage, wall_motion_storage, pressure_storage;
    WallPhases3D wall_phases;
    // Velocity-driven groups need their accepted pose and rates after scratch eviction.
    Kokkos::View<const SphBoundary::SampledWallMotion3D*, Kokkos::HostSpace> wall_motion;
    Geometry3D geometry;
    FieldStamp stamp;
    double maximum_step{};
    std::optional<ReadField<Values::const_type>> pressure;
    IntegrationHistory3D history;
};

enum class FieldGroup3D { density, pressure, velocity };
struct StepOutputs3D {
    Values density, pressure;
    Acceleration3D acceleration;
    Velocity3D velocity;
};
using FieldWrite3D = std::function<Status(const StepOutputs3D&)>;
using BindFields3D = std::function<Status(FieldGroup3D, const FieldWrite3D&,
                                          const std::function<bool()>& evict_unused)>;

class WcsphStep3D {
  public:
    WcsphStep3D(const Kokkos::Serial&, const detail::ResourceBudget&, std::size_t max_cells,
                WcsphVerletParameters);
    Status initialize_state(const ParticleFields3D&, const Velocity3D::const_type&,
                            const Values::const_type&, const std::vector<SphBoundary::WallGroupInput3D>&,
                            WcsphState3D&, const BindFields3D&);
    // ParticipantIF::step admits a finite positive duration within the published CFL/wall limit.
    Status try_step(const TimeInterval&, const WcsphState3D& current, WcsphState3D& candidate,
                    const BindFields3D&);
    void release_workspace() noexcept;

  private:
    friend struct WcsphTestAccess;
    friend class WcsphParticipant3D;
    using Ticket = detail::ResourceBudget::Ticket;
    template <class V> struct Buffer {
        Ticket ticket;
        V values;
    };
    /// Reusable position storage. A reordered candidate may take ownership until its geometry is released.
    std::optional<Buffer<Velocity3D>> trial_positions_;
    bool release_position_buffer() noexcept;
    Result<Ticket> reserve_storage(std::size_t bytes);

    const Kokkos::Serial& exec_;
    const detail::ResourceBudget& budget_;
    std::size_t max_cells_;
    WcsphVerletParameters parameters_;
    Ticket wall_storage_;
    struct WallGroup {
        std::uint64_t id;
        std::size_t phase_index;
        std::variant<SphBoundary::FixedWall3D, SphBoundary::PrescribedMovingWall3D> boundary;
        const SphBoundary::SolidWallBoundary3D& source() const;
        const SphBoundary::PrescribedMovingWall3D* moving() const;
    };
    using WallMotions = Buffer<Kokkos::View<SphBoundary::SampledWallMotion3D*, Kokkos::HostSpace>>;
    std::vector<WallGroup> walls_;
    std::size_t wall_count_{}, moving_groups_{};
    bool has_velocity_driven_particles_{};
    void capture_groups(const std::vector<SphBoundary::WallGroupInput3D>&);
    WallPhases3D::non_const_type phase_storage(WcsphState3D&);
    WallMotions sample_wall_motion(WcsphState3D& candidate, SphBoundary::WallMotionUse3D,
                                   const WcsphState3D* accepted = nullptr, real dt = 0);
    void set_next_step(WcsphState3D&, real largest_speed_squared) const;
    struct MovingWallParticle {
        std::size_t phase_index;
        const std::array<real, 3>& reference_position;
    };
    std::optional<MovingWallParticle> moving_wall_particle(std::uint64_t particle_id) const;
    void write_wall_positions(const ParticleFields3D&, const WallMotions&, const Velocity3D& output) const;
    SphBoundary::WallTrajectoryTime3D committed_wall_time(std::uint64_t particle_id, const WallPhases3D&) const;

    std::unique_ptr<CellGrid> committed_grid_;

    WallMotions advance_wall_motion(const WcsphState3D&, double duration, WcsphState3D&);
    std::unique_ptr<CellGrid> advance_positions_and_update_neighbors(
        const WcsphState3D&, real dt, const WallMotions&, WcsphState3D&);
    real advance_density_and_velocity(const WcsphState3D&, double duration, WcsphState3D&, const BindFields3D&);
    Buffer<Geometry3D::Indices::non_const_type> fluid_rows_by_slot(const CellGrid&);
    void evaluate_density_rate(WcsphState3D&, const CellGrid&, const WallMotions&);
    void evaluate_pressure_acceleration(WcsphState3D&, const CellGrid&, const BindFields3D&, const WallMotions&,
                                        bool initial_density = false);
};
} // namespace nexora::sph

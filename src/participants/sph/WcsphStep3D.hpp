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

struct WcsphState3D {
    detail::ResourceBudget::Ticket phase_storage;
    WallPhases3D wall_phases;
    Geometry3D geometry;
    FieldStamp stamp;
    double maximum_step{};
    std::array<detail::ResourceBudget::Ticket, 3> field_storage;
    std::optional<ReadField<Values::const_type>> density, pressure;
    std::optional<ReadField<Acceleration3D::const_type>> acceleration, velocity;
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
enum class StepStage3D { initial, start, end };

class WcsphStep3D {
  public:
    WcsphStep3D(const Kokkos::Serial&, const detail::ResourceBudget&, std::size_t max_cells,
                WcsphVerletParameters);
    Status initialize_state(const ParticleFields3D&, const Velocity3D::const_type&,
                            const Values::const_type&, const std::vector<SphBoundary::WallGroupInput3D>&,
                            WcsphState3D&, const BindFields3D&);
    Status try_step(const TimeInterval&, const WcsphState3D& current, WcsphState3D& candidate,
                    const BindFields3D&);
    void release_workspace() noexcept;

  private:
    friend struct WcsphTestAccess;
    friend class WcsphParticipant3D;
    void* observer_context_{};
    void (*observe_)(void*, StepStage3D, const NeighborView3D&){};
    using Ticket = detail::ResourceBudget::Ticket;
    template <class V> struct Buffer {
        Ticket ticket;
        V values;
    };
    struct Workspace {
        std::optional<Buffer<Velocity3D>> half, stage;
        std::optional<Buffer<Geometry3D::Indices::non_const_type>> old_row;
        std::optional<Buffer<DensityRate3D>> rate_n, rate_star;
        std::optional<Buffer<Values>> corrected;

        bool release_rates() noexcept;
        bool release_continuity() noexcept;
    };
    const Kokkos::Serial& exec_;
    const detail::ResourceBudget& budget_;
    std::size_t max_cells_;
    WcsphVerletParameters parameters_;
    Ticket wall_storage_;
    struct WallGroup {
        std::uint64_t id;
        std::size_t sample_offset, phase_index;
        std::variant<SphBoundary::FixedWall3D, SphBoundary::PrescribedMovingWall3D> boundary;
        const SphBoundary::SolidWallBoundary3D& source() const;
        const SphBoundary::PrescribedMovingWall3D* moving() const;
    };
    using WallSamples = Buffer<Kokkos::View<SphBoundary::WallKinematics3D*, Kokkos::HostSpace>>;
    std::vector<WallGroup> walls_;
    std::size_t wall_count_{}, moving_count_{}, moving_groups_{};
    void capture_groups(const std::vector<SphBoundary::WallGroupInput3D>&, std::size_t fluid_count);
    Ticket reserve_wall_stage(std::size_t bytes);
    WallPhases3D::non_const_type phase_storage(WcsphState3D&);
    WallSamples sample_groups(const WallPhases3D&);
    void set_next_step(WcsphState3D&) const;
    SphBoundary::WallKinematics3D sample_at(const WallGroup&, const SphBoundary::WallParticle3D&,
                                           const WallSamples&) const;
    template <class Write>
    void write_wall_slots(const ParticleFields3D&, const WallSamples&, const char*, const Write&) const;
    Result<SphBoundary::WallTrajectoryTime3D> wall_time(std::uint64_t particle_id, const WallPhases3D&) const;

    std::unique_ptr<CellGrid> committed_grid_;
    Workspace workspace_;

    void evaluate_pressure(WcsphState3D&, const CellGrid&, const BindFields3D&, Workspace*, const WallSamples&);
};
} // namespace nexora::sph
